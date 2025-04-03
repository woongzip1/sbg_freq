import os
import sys
import random
from typing import List
import numpy as np
import torch
import torch.utils.data
import torchaudio as ta
import torchaudio.functional as aF

def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):
    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    log_amp = torch.log(torch.abs(stft_spec)+1e-4)
    pha = torch.angle(stft_spec)

    com = torch.stack((torch.exp(log_amp)*torch.cos(pha), 
                       torch.exp(log_amp)*torch.sin(pha)), dim=-1)

    return log_amp, pha, com


def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
    amp = torch.exp(log_amp)
    com = torch.complex(amp*torch.cos(pha), amp*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return audio

def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    """ Get list of all audio paths """
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files 
                            if os.path.splitext(file)[-1].lower() in file_extensions] 
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def get_filename(path):
    return os.path.splitext(os.path.basename(path))

def make_dataset(config, mode:str):
    return Dataset(
        **config.dataset.common,
        **config.dataset[mode],
        mode=mode,
    )
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_dir_nb: List[str],
                 path_dir_wb: List[str],
                 seg_len: float = 0.9,
                 sr: int = 48000,
                 mode: str = "train",
                 ):
        assert isinstance(path_dir_nb, list), "PATH must be a list"

        self.seg_len = seg_len
        self.mode = mode
        self.sr = sr        
        paths_wav_wb = []
        paths_wav_nb = []
    
        # number of datasets -> ['path1','path2']
        for i in range(len(path_dir_nb)):
            self.path_dir_nb = path_dir_nb[i]
            self.path_dir_wb = path_dir_wb[i]

            wb_files = get_audio_paths(self.path_dir_wb, file_extensions='.wav')
            nb_files = get_audio_paths(self.path_dir_nb, file_extensions='.wav')
            paths_wav_wb.extend(wb_files)
            paths_wav_nb.extend(nb_files)

            print(f"Index:{i} with {len(wb_files)} samples")

        if len(paths_wav_wb) != len(paths_wav_nb):
            raise ValueError(f"Error: LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers are different!")

        # make filename pairs: wb-nb        
        self.filenames = list(zip(paths_wav_wb, paths_wav_nb))
        print(f"LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers loaded!")

    def _multiple_pad(self, wav, N=80):
        pad_len = (N - wav.shape[-1] % N) % N
        wav = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0)
        return wav

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        N = 80
        path_wav_wb, path_wav_nb = self.filenames[idx]

        wav_nb, sr_nb = ta.load(path_wav_nb)
        wav_wb, sr_wb = ta.load(path_wav_wb)

        wav_wb = wav_wb.view(1, -1)
        wav_nb = wav_nb.view(1, -1)

        if self.mode == "train":
            duration = int(self.seg_len * self.sr) # 43200
            duration = (duration // N) * N # multiple of N
            
            if wav_nb.shape[-1] < duration:
                wav_nb = self.ensure_length(wav_nb, duration)
                wav_wb = self.ensure_length(wav_wb, duration)
            elif wav_nb.shape[-1] > duration:
                start_idx = np.random.randint(0, wav_nb.shape[-1] - duration)
                wav_nb = wav_nb[:, start_idx:start_idx + duration]
                wav_wb = wav_wb[:, start_idx:start_idx + duration]

        elif self.mode == "val": 
            wav_nb = self._multiple_pad(wav_nb)
            wav_wb = self._multiple_pad(wav_wb)            
        else:
            sys.exit(f"unsupported mode! (train/val)")
        return wav_wb, wav_nb, get_filename(path_wav_wb)[0]

    @staticmethod
    def ensure_length(wav, target_length):
        target_length = int(target_length)
        if wav.shape[1] < target_length:
            pad_size = target_length - wav.shape[1]
            wav = F.pad(wav, (0, pad_size))
        elif wav.shape[1] > target_length:
            wav = wav[:, :target_length]
        return wav
        
    def set_maxlen(self, wav, max_lensec):
        sr = self.sr
        max_len = int(max_lensec * sr)
        if wav.shape[1] > max_len:
            # print(wav.shape, max_len)
            wav = wav[:, :max_len]
        return wav
    def __len__(self):
        return len(self.filenames)

if __name__ == "__main__":
    from main import load_config
    config_path = 'configs/config_template.yaml'
    config = load_config(config_path)
    train_dataset = make_dataset(config, 'train')
    print(len(train_dataset))