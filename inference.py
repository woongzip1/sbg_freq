import torch
import yaml
import pdb
import gc
import warnings
import time
import random
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler

import soundfile as sf
from tqdm import tqdm
from box import Box

from utils import *

## models
from models.prepare_models import MODEL_MAP, prepare_discriminator, prepare_generator
## dataset
from dataset import Dataset, make_dataset, amp_pha_istft, amp_pha_stft
from trainer import Trainer
## main utils
from main import load_config, prepare_dataloader

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_model_params(model, checkpoint_path, device='cuda'):
    model = model.to(device)
    print(f"Loading '{checkpoint_path}...'")
    ckpt = torch.load(checkpoint_path)
    # import pdb
    # pdb.set_trace()
    # model.load_state_dict(ckpt['generator'])
    model.load_state_dict(ckpt['generator_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model

def lpf(y, sr=16000, cutoff=500, numtaps=4096, window='hamming', figsize=(10,2)):
    from scipy.signal import firwin, lfilter, freqz
    """ 
    Applies FIR filter
    cutoff freq: cutoff freq in Hz
    """
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    taps = firwin(numtaps=numtaps, cutoff=normalized_cutoff, window=window)
    y_lpf = lfilter(taps, 1.0, y)
    # y_lpf = np.convolve(y, taps, mode='same')
    
    # Length adjust
    y_lpf = np.roll(y_lpf, shift=-numtaps//2)
    return y_lpf

def _pad_signal(waveform:torch.tensor, multiple_factor:int):
    import torch.nn.functional as F
    pad_len = (multiple_factor - waveform.shape[-1] % multiple_factor) % multiple_factor
    if pad_len > 0:
        waveform = F.pad(waveform, (0, pad_len))
    return waveform, pad_len
    
def _generate_target_signal(lr, hr, cutoff_freq=4000, sr=48000, n_fft=1536, hop_length=768):
    # Connsider this setting is valid ?
    lr, pad_len = _pad_signal(lr, multiple_factor=n_fft)
    hr, pad_len = _pad_signal(hr, multiple_factor=n_fft)
    
    # Maintain narrow band spectrum
    hann_window = torch.hann_window(n_fft).to(lr.device)
    lr_spec = torch.stft(lr, n_fft=n_fft, hop_length=hop_length, window=hann_window, return_complex=True)
    hr_spec = torch.stft(hr, n_fft=n_fft, hop_length=hop_length, window=hann_window, return_complex=True)
    freq_bin = int((cutoff_freq / sr) * n_fft) 

    new_spec = torch.cat([
        lr_spec[:,:freq_bin, :],  # low freq from LR
        hr_spec[:,freq_bin:, :]   # high freq from HR
    ], dim=1) # [B,F,T]

    # log_amp = torch.log1p(torch.abs(new_spec))
    log_amp = torch.log(torch.abs(new_spec) + 1e-4)
    phase = torch.angle(new_spec)
    spectrum = {
            'log_amp': log_amp,
            'phase': phase
        }
    # pdb.set_trace()
    target_waveform = torch.istft(new_spec, n_fft=n_fft, hop_length=hop_length,  window=hann_window, length=hr.shape[-1])
    target_waveform = target_waveform[...,:-pad_len]
    return target_waveform, spectrum

def _forward_pass(lr_waveform, hr_waveform=None, generator=None, config=None):
    if config.generator.type == "seanet":
        audio_wb_g = generator(lr_waveform)
        spectrum = {}
        loss_dict = {}
    elif config.generator.type == "resnet_apbwe":
        audio_wb_g, spectrum, loss_dict = generator(lr_waveform, hr_waveform)
    else:
        mag_nb, pha_nb, _ = amp_pha_stft(lr_waveform.squeeze(1), config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
        mag_wb_g, pha_wb_g, com_wb_g = generator(mag_nb, pha_nb)
        audio_wb_g = amp_pha_istft(mag_wb_g, pha_wb_g, config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
        spectrum = {
            'log_amp': mag_wb_g,
            'phase': pha_wb_g
            }
        loss_dict = {}
    return audio_wb_g, spectrum, loss_dict


def inference(config, device='cuda', save_lr=False, save_target=False, save_lpf_signals=False, exp_name=''):
    # save_base_dir = os.path.join(config['inference']['dir_speech'], exp_name)
    # save_base_dir = os.path.join(config['inference']['dir_audio'], exp_name)
    # dataloader
    dataloaders = prepare_dataloader(config) # train, val, val_speech
    dirs_dict = config['inference']
    for i, (_, save_dir) in enumerate(dirs_dict.items()):
        print('***', os.path.basename(save_dir), '***')
        save_base_dir = os.path.join(save_dir, exp_name)
        os.makedirs(save_base_dir, exist_ok=True)
        
        # Dataloader
        val_loader = dataloaders[i+1]
        
        # generator
        model = prepare_generator(config, MODEL_MAP)
        model = load_model_params(model, config['train']['ckpt_path'], device=device)
        
        # pdb.set_trace()
        
        set_seed()
        ## forward
        model.eval()
        bar = tqdm(val_loader)
        duration_tot = 0
        with torch.no_grad():
            for batch in bar:
                hr, lr, name = batch[0].to(device), batch[1].to(device), batch[2]

                # forward
                pred_start = time.time() # tick
                audio_gen, _, _ = _forward_pass(lr, hr, model, config)
                # mag_nb, pha_nb, com_nb = amp_pha_stft(lr.squeeze(0), config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
                # mag_g, pha_g, com_g = model(mag_nb, pha_nb)
                # audio_gen = amp_pha_istft(mag_g, pha_g, config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
                duration_tot += time.time() - pred_start # tock
                
                # high target
                if save_target:
                    target, spectrum_r = _generate_target_signal(lr.squeeze(1), hr.squeeze(1))
                    # target = lpf(target.squeeze().cpu().numpy(), sr=48000, cutoff=11000)
                    target_file = os.path.join(save_base_dir, 'target_lpf', name[0]+'.wav')
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    sf.write(target_file, target.squeeze().cpu().numpy(), 48000, 'PCM_16')
                
                if save_lpf_signals:
                    lpf_signal = lpf(audio_gen.squeeze().cpu().numpy(), sr=48000, cutoff=11000)
                    target_file = os.path.join(save_base_dir, 'lpf_signal', name[0]+'.wav')
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    sf.write(target_file, lpf_signal.squeeze(), 48000, 'PCM_16')
                
                # save
                output_file = os.path.join(save_base_dir, name[0]+'.wav')
                # sf.write(output_file, audio_gen.squeeze().cpu().numpy(), 48000, 'PCM_16')

                if save_lr:
                    lr_file = os.path.join(save_base_dir, 'lr', name[0]+'.wav')
                    os.makedirs(os.path.dirname(lr_file), exist_ok=True)
                    sf.write(lr_file, lr.squeeze().cpu().numpy(), 48000, 'PCM_16')
            print(f'duration_tot!:{duration_tot}')
        
def main():
    print("Initializing Inference Process...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--exp_name", type=str, default='') 
    parser.add_argument("--save_lr", type=bool, default=False)
    parser.add_argument("--save_target", type=bool, default=False)
    parser.add_argument("--save_lpf_signals", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()
    
    config = load_config(args.config)

    inference(config, device=args.device, 
              save_lr=args.save_lr, save_target=args.save_target, save_lpf_signals=args.save_lpf_signals,
              exp_name=args.exp_name)

if __name__ == "__main__":
    main()