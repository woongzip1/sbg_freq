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

def _forward_pass(lr_waveform, hr_waveform, generator, config,):
    if config.generator.type == "seanet":
        audio_wb_g = generator(lr_waveform)
    elif config.generator.type == 'resnet_apbwe':
        audio_wb_g,_,_ = generator(lr_waveform, hr_waveform)
    else:
        mag_nb, pha_nb, _ = amp_pha_stft(lr_waveform.squeeze(1), config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
        mag_wb_g, pha_wb_g, com_wb_g = generator(mag_nb, pha_nb)
        audio_wb_g = amp_pha_istft(mag_wb_g, pha_wb_g, config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
    return audio_wb_g


def inference(config, device='cuda', save_lr=False, exp_name=''):
    # save_base_dir = os.path.join(config['inference']['dir_speech'], exp_name)
    # save_base_dir = os.path.join(config['inference']['dir_audio'], exp_name)
    
    dirs_dict = config['inference']
    for _, save_dir in dirs_dict.items():
        print('***', os.path.basename(save_dir), '***')
        save_base_dir = os.path.join(save_dir, exp_name)
        os.makedirs(save_base_dir, exist_ok=True)

        # dataloader
        _, val_loader = prepare_dataloader(config)
        # generator
        model = prepare_generator(config, MODEL_MAP)
        model = load_model_params(model, config['train']['ckpt_path'], device=device)
        
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
                audio_gen = _forward_pass(lr, hr, model, config)
                # mag_nb, pha_nb, com_nb = amp_pha_stft(lr.squeeze(0), config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
                # mag_g, pha_g, com_g = model(mag_nb, pha_nb)
                # audio_gen = amp_pha_istft(mag_g, pha_g, config.stft.n_fft, config.stft.hop_size, config.stft.win_size)
                duration_tot += time.time() - pred_start # tock
                
                # save
                output_file = os.path.join(save_base_dir, name[0]+'.wav')
                sf.write(output_file, audio_gen.squeeze().cpu().numpy(), 48000, 'PCM_16')

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
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()
    
    config = load_config(args.config)

    inference(config, device=args.device, save_lr=args.save_lr, exp_name=args.exp_name)

if __name__ == "__main__":
    main()