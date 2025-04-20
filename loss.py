import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.signal.windows import hann
# from soundstream.balancer import *


## To Do: loss into dictionary ...

class LossCalculator:
    def __init__(self, config, discriminator):
        self.discriminator = discriminator
        self.ms_mel_loss_config = config['loss']['ms_mel_loss_config']
        
        self.lambda_loss = {
            'adv': config['loss'].get('lambda_adv_loss', 0),
            'fm': config['loss'].get('lambda_fm_loss', 0),
            'mel': config['loss'].get('lambda_mel_loss', 0),
            'codebook': config['loss'].get('lambda_codebook_loss', 0),
            'commit': config['loss'].get('lambda_commit_loss', 0),
            'phase': config['loss'].get('lambda_phase_loss', 0),
            'consistency': config['loss'].get('lambda_consistency_loss', 0),
            'logamp': config['loss'].get('lambda_logamp_loss', 0),
        }
        
    def _save_figures(self, hr, x_hat_full):
        ## for debugging
        hr = hr[0,...]
        x_hat_full = x_hat_full[0,...]
        from matplotlib import pyplot as plt
        # clip front samples
        plt.figure(figsize=(14, 4))
        plt.plot(hr.squeeze().cpu().detach().numpy(), label='gt')
        plt.plot(x_hat_full.squeeze().cpu().detach().numpy() + 0.1, label='recon')
        # start = x.shape[-1] - num_taps
        start = 0
        step = self.num_taps + 200
        plt.xlim(start, start + step)
        plt.legend()
        # Save the figure instead of showing it
        save_path = "output_plot.png"  # Define the output file path
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_generator_loss(self, hr, x_hat_full, **kwargs): 
        hr = hr.squeeze(1)
        x_hat_full = x_hat_full.squeeze(1)

        assert hr.shape == x_hat_full.shape, f"Size mismatch: {hr.shape}, {x_hat_full.shape}"

        # adversarial & feature matching loss
        g_loss_dict, g_loss_report = self.discriminator.g_loss(hr, x_hat_full, adv_loss_type='hinge')

        # commit & codebook loss
        commit_loss = kwargs.get('commit_loss', 0)
        codebook_loss = kwargs.get('codebook_loss', 0)
        
        # mel loss
        ms_mel_loss_value = (
            ms_mel_loss(hr, x_hat_full, **self.ms_mel_loss_config)
            if self.lambda_loss['mel'] > 0 else 0   
        )
        
        # stft consistency loss
        consistency_loss = (
            stft_consistency_loss(wav=x_hat_full, 
                                  stft_from_model=kwargs.get('stft_from_model', None), 
                                  config=kwargs.get('config', None))    
            if self.lambda_loss['consistency'] > 0 else 0
        )
        # logamp loss
        logamp_loss_val = (
                logamp_loss(
                    logamp_r=kwargs.get('logamp_r', None),
                    logamp_g=kwargs.get('logamp_g', None)
                ) if self.lambda_loss['logamp'] > 0 else 0
        )
        # phase loss
        if self.lambda_loss['phase'] > 0:
            phase_r = kwargs.get('phase_r', None)
            phase_g = kwargs.get('phase_g', None)
            import pdb
            # pdb.set_trace()
            
            phase_loss_dict = phase_losses(phase_r=phase_r, phase_g=phase_g)
            phase_loss = phase_loss_dict['total']
        else:
            phase_loss_dict = {}
            phase_loss = 0
            
        # Generator loss
        loss_G = (
            self.lambda_loss['adv'] * g_loss_dict.get('adv_g', 0) +
            self.lambda_loss['fm'] * g_loss_dict.get('fm', 0) +
            self.lambda_loss['mel'] * ms_mel_loss_value +
            self.lambda_loss['codebook'] * codebook_loss +
            self.lambda_loss['commit'] * commit_loss +
            self.lambda_loss['phase'] * phase_loss + 
            self.lambda_loss['consistency'] * consistency_loss +
            self.lambda_loss['logamp'] * logamp_loss_val
        )

        loss_dict = {
            'total': loss_G,
            'mel': ms_mel_loss_value,
            'adv': g_loss_dict.get('adv_g', 0),
            'fm': g_loss_dict.get('fm', 0),
            'codebook': codebook_loss,
            'commit': commit_loss,
            'phase': phase_loss,
            'phase_dict': phase_loss_dict,
            'consistency': consistency_loss,
            'logamp': logamp_loss_val,
            'g_loss_dict': g_loss_dict,
            'g_report': g_loss_report,
        }
        
        # return loss_G, ms_mel_loss_value, g_loss_dict, g_loss_report
        return loss_dict

    def compute_discriminator_loss(self, hr, x_hat_full):
        d_loss_dict, d_loss_report = self.discriminator.d_loss(hr, x_hat_full, adv_loss_type='hinge')
        loss_D = d_loss_dict.get('adv_d', 0)
        # loss_D = sum(d_loss_dict.values())
        # loss_dict = {
        #     'total': loss_D,
        #     'd_loss_dict': d_loss_dict,
        #     'd_report': d_loss_report,
        # }
        return loss_D, d_loss_dict, d_loss_report
        # return loss_dict

def wav_to_stft(x, config):
    config = config.generator.decoder_cfg.stft
    # pad
    pad_len = (config.n_fft - x.shape[-1] % config.n_fft) % config.n_fft
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    complex_stft = torch.stft(x, window=torch.hann_window(config.n_fft).to(x.device), return_complex=False, **config)
    return complex_stft

def stft_consistency_loss(wav, stft_from_model, config):
    stft_from_wav = wav_to_stft(wav, config)
    import pdb
    # pdb.set_trace()
    # sc_loss = torch.norm(T - S, p='fro') / (torch.norm(T, p='fro') + 1e-7)
    # import pdb
    # pdb.set_trace()

    ## sample stft value > 1000 ...
    consistency_loss = F.mse_loss(stft_from_wav[:,:,:-1,:], stft_from_model[:,:,:-1,:])
    return consistency_loss

# def stft_consistency_loss(wav, stft_from_model, config, eps=1e-7):
#     T = wav_to_stft(wav, config)  # complex tensor [B, F, T]
#     S = stft_from_model
#     T_r, T_i = T[...,0], T[...,1]
#     S_r, S_i = S[...,0], S[...,1]
#     sc_r = torch.norm(T_r - S_r, p='fro') / (torch.norm(T_r, p='fro') + eps)
#     sc_i = torch.norm(T_i - S_i, p='fro') / (torch.norm(T_i, p='fro') + eps)
#     return 0.5 * (sc_r + sc_i)


def logamp_loss(logamp_r, logamp_g):
    return F.mse_loss(logamp_r, logamp_g)


def phase_losses(phase_r, phase_g):
    """
    phase_r: real data
    phase_g: generated data
    """
    phase_loss_dict = {}
    # print('phase shape:', phase_r.shape)
    # import pdb
    # pdb.set_trace()
    phase_loss_dict['ip_loss'] = torch.mean(anti_wrapping_function(phase_r - phase_g))
    phase_loss_dict['gd_loss'] =  torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    phase_loss_dict['iaf_loss'] = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    # return ip_loss, gd_loss, iaf_loss
    phase_loss_dict['total'] = (
        phase_loss_dict['ip_loss'] +
        phase_loss_dict['gd_loss'] +
        phase_loss_dict['iaf_loss']
    )
    return phase_loss_dict 

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

    
def compute_subband_loss(hf_estimate: torch.Tensor, target_subbands: torch.Tensor):
    subband_loss, sc_loss, mag_loss = ms_stft_loss(x=target_subbands, x_hat=hf_estimate)
    return subband_loss

def ms_stft_loss(x, x_hat, n_fft_list=[16, 32, 64], hop_ratio=0.25, eps=1e-5, freq_range=None):
    """
    reference: https://github.com/AppleHolic/multiband_melgan/
    
    Multi-Scale STFT Loss with Spectral Convergence and Magnitude Loss
    Args:
        x (torch.Tensor) [B, T]: ground truth waveform
        x_hat (torch.Tensor) [B, T]: generated waveform
        n_fft_list (List of int): list of n_fft for each scale
        hop_ratio (float): hop_length = n_fft * hop_ratio
        sr (int): sampling rate
        eps (float): epsilon for numerical stability in log calculation
    Returns:
        total_loss: combined STFT loss (SC + Magnitude)
        sc_loss: Spectral Convergence loss
        mag_loss: Magnitude loss
    """
    assert len(n_fft_list) > 0, "n_fft_list must contain at least one value."
    assert x.shape == x_hat.shape, "input and target must have the same shape"
    
    total_loss, sc_loss, mag_loss = 0., 0., 0.
    
    if len(x.shape) == 3: # subband input
        assert x.shape[1] > 1, "Expected Subband Input with multiple channels, got shape: {}".format(x.shape)
        B,C,_ = x.shape
        x = x.reshape(B*C, -1) # [B,C,T] -> [BC,T]
        x_hat = x_hat.reshape(B*C, -1)
    else: # fullband input
        C = 1
        
    for n_fft in n_fft_list:
        sig_to_spg = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=int(n_fft * hop_ratio), # magnitude
                                   power=1.0, normalized=False, center=True).to(x.device)

        # Compute spectrograms for x and x_hat
        spg_x = sig_to_spg(x)  # [B, F, T']
        spg_x_hat = sig_to_spg(x_hat)  # [B, F, T']
        # print(spg_x.shape)
        
        # 
        if freq_range is not None:
            f_low = math.ceil(n_fft//2 * freq_range[0])
            f_high =  math.ceil(n_fft//2 * freq_range[1])
            spg_x = spg_x[:,f_low:f_high,:]
            spg_x_hat = spg_x_hat[:,f_low:f_high,:]
        
        # SC, Mag loss
        sc_loss_ = ((spg_x - spg_x_hat).norm(p='fro', dim=(1, 2)) / spg_x.norm(p='fro', dim=(1, 2)) + eps).mean()
        mag_loss_ = torch.mean(torch.abs(torch.log(spg_x.clamp(min=eps)) - torch.log(spg_x_hat.clamp(min=eps))))
        
        # Accumulate losses
        sc_loss += sc_loss_
        mag_loss += mag_loss_
        total_loss += sc_loss_ + mag_loss_
    
    scale_factor = len(n_fft_list)
    return total_loss/scale_factor, sc_loss/scale_factor, mag_loss/scale_factor
    
def ms_mel_loss(x, x_hat, n_fft_list=[32, 64, 128, 256, 512, 1024, 2048], hop_ratio=0.25, 
                mel_bin_list=[5, 10, 20, 40, 80, 160, 320], fmin=0, fmax=None, sr=48000, mel_power=1.0,
                core_cutoff = 4500,
                eps=1e-5, reduction='mean', **kwargs):
    """
    Multi-scale spectral energy distance loss
    References:
        Kumar, Rithesh, et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS, 2023.
    Args:
        x (torch.Tensor) [B, ..., T]: ground truth waveform (Batched)
        x_hat (torch.Tensor) [B, ..., T]: generated waveform
        n_fft_list (List of int): list that contains n_fft for each scale
        hop_ratio (float): hop_length = n_fft * hop_ratio
        mel_bin_list (List of int): list that contains the number of mel bins for each scale
        sr (int): sampling rate
        fmin (float): minimum frequency for mel-filterbank calculation
        fmax (float): maximum frequency for mel-filterbank calculation
        mel_power (float): power to raise magnitude to before taking log
    Returns:
    """
    assert len(n_fft_list) == len(mel_bin_list)
    assert x.shape == x_hat.shape, "input and target must have the same shape"
    
    loss = 0
    for n_fft, mel_bin in zip(n_fft_list, mel_bin_list):
        sig_to_spg = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=int(n_fft * hop_ratio), 
                                    window_fn=hann, wkwargs={"sym": False},\
                                    power=1.0, normalized=False, center=True).to(x.device)
        spg_to_mel = T.MelScale(n_mels=mel_bin, sample_rate=sr, n_stft=n_fft//2+1, f_min=fmin, f_max=fmax, norm="slaney", mel_scale="slaney").to(x.device)  
        
        ## erase core band considerationsx.shap
        # x_spg = sig_to_spg(x) # [B,F,T]
        # f_start = int((core_cutoff / sr) * n_fft) + 1
        # x_spg[:,:f_start,:] = 0
        # x_mel = spg_to_mel(x_spg) # [B, C, mels, T]
        
        # x_hat_spg = sig_to_spg(x_hat)
        # x_hat_spg[:,:f_start,:] = 0
        # x_hat_mel = spg_to_mel(x_hat_spg)
        #####
        x_mel = spg_to_mel(sig_to_spg(x))  # [B, C, mels, T]
        x_hat_mel = spg_to_mel(sig_to_spg(x_hat))

        log_term = torch.sum(torch.abs(x_mel.clamp(min=eps).pow(mel_power).log10() - x_hat_mel.clamp(min=eps).pow(mel_power).log10()))
        if reduction == 'mean':
            log_term /= torch.numel(x_mel)
        elif reduction == 'sum':
            log_term /= x_mel.shape[0]
        
        loss += log_term
        
    return loss