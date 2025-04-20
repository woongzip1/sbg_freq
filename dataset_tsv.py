import pdb
import os
import sys
import random
import numpy as np
import torch
import torchaudio as ta
from pathlib import Path
from typing import List

## supports global multisampling [12, 16, 20 kbps]

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
    return IndexedTSVDataset(
        **config.dataset.common,
        **config.dataset[mode],
        mode=mode,
    )
    
import random
import torch, torchaudio as ta
import numpy as np
from pathlib import Path

def read_tsv_firstcol(tsv_path):
    with open(tsv_path) as f:
        return [line.split('\t',1)[0].strip() for line in f if line.strip()]

class IndexedTSVDataset(torch.utils.data.Dataset):
    def __init__(self,
                 wb_tsv: str,
                 nb_tsv_list,
                 seg_len=0.9,
                 sr=48000,
                 mode="train"):
        self.seg_len, self.sr, self.mode = seg_len, sr, mode

        # 1) read tsvs
        self.wb_paths = read_tsv_firstcol(wb_tsv)
        print(len(self.wb_paths), 'samples loaded!')
        self.nb_lists = [read_tsv_firstcol(p) for p in nb_tsv_list]
        for i,lists in enumerate(self.nb_lists):
            print(len(lists), f'samples for index {i}')
            
        # sanity
        L = len(self.wb_paths)
        assert all(len(lst)==L for lst in self.nb_lists), "TSV line counts differ"

    def __len__(self):
        return len(self.wb_paths)

    def _pad(self, wav, N=80):
        pad = (N - wav.shape[-1] % N) % N
        return torch.nn.functional.pad(wav, (0,pad))

    def _ensure(self, wav, L):
        if wav.shape[-1] < L: 
            wav = torch.nn.functional.pad(wav, (0, 4000)) # offset
            reps = (L + wav.shape[-1] - 1) // wav.shape[-1]          # ceil(L / wav.shape[-1])
            wav = wav.repeat(1, reps)[..., :L]   # repeat
        elif wav.shape[-1] > L:
            wav = wav[..., :L]
        return wav

    def __getitem__(self, idx):
        wb_path = self.wb_paths[idx]

        # pick one nb tsv list randomly
        bit_idx = random.randrange(len(self.nb_lists))     # 0,1,2 → 12/16/20 kbps
        nb_path = self.nb_lists[bit_idx][idx]
        # nb_row = random.choice(self.nb_lists)[idx]

        wav_wb, _ = ta.load(wb_path); 
        wav_nb, _ = ta.load(nb_path);  

        if self.mode=="train":
            N=80
            target_signal_len = int(self.seg_len*self.sr)//N*N # multiple of N 
            current_signal_len = min(wav_wb.shape[-1], wav_nb.shape[-1]) # min len
            if current_signal_len <= target_signal_len:
                wav_wb = self._ensure(wav_wb[..., :current_signal_len], target_signal_len)
                wav_nb = self._ensure(wav_nb[..., :current_signal_len], target_signal_len)
            else:
                s = np.random.randint(0, current_signal_len-target_signal_len)
                wav_wb = wav_wb[..., s:s+target_signal_len]
                wav_nb = wav_nb[..., s:s+target_signal_len]
        elif self.mode in ['val','val_speech']:
            wav_wb = self._pad(wav_wb)
            wav_nb = self._pad(wav_nb)
        else:
            sys.exit(f"unsupported mode! (train/val)") 
        return wav_wb, wav_nb, Path(wb_path).stem, bit_idx

if __name__ == "__main__":
    from main import load_config
    config_path = 'configs/config_template.yaml'
    config = load_config(config_path)
    train_dataset = make_dataset(config, 'train')
    print(len(train_dataset))
    for i in train_dataset:
        print(len(i))
        wb,nb,name,bitidx = i
        print(wb.shape, nb.shape, name, bitidx)
        pdb.set_trace()
        # break
    