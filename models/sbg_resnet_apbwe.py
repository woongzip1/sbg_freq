import pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from models.model import APNet_BWE_Model
from models.resnet_encoder import ResnetEncoder
from models.quantize import ResidualVectorQuantize
from main import load_config

class ResNet_APBWE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h = config.generator
        self.encoder = self._build_encoder(self.h.encoder_cfg)       
        self.quantizer = self._build_quantizer(self.h.quantizer_cfg)
        self.decoder = self._build_decoder(self.h.decoder_cfg)
        self.apply(self._init_weights) 
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            # nn.init.xavier_uniform_(m.weight)
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)            
    
    def _compute_stft(self, waveform, n_fft=2048, center=True,
                      win_length=2048, hop_length=2048, pad=True, **kwargs):
        
        # pad len
        if pad:
            padlen = (n_fft - waveform.shape[-1] % n_fft) % n_fft
            waveform = F.pad(waveform, (0, padlen))
        
        if waveform.dim() == 3: # [B,1,L] -> [B,L]
            waveform = waveform.squeeze(1)
            
        hann_window = torch.hann_window(win_length).to(waveform.device)
        stft_spec = torch.stft(waveform, n_fft, hop_length=hop_length, win_length=win_length, window=hann_window, 
                            center=center, pad_mode='reflect', normalized=False, return_complex=True) # [B,F(complex),T]
        log_amp = torch.log(torch.abs(stft_spec)+1e-4)  # [B,F(log),T]
        pha = torch.angle(stft_spec)                    # [B,F,T] 
        com = torch.stack((torch.exp(log_amp)*torch.cos(pha),
                        torch.exp(log_amp)*torch.sin(pha)), dim=-1) # [B,F,T,2(real-im)]

        # note original mbseanet used log-power spectra & normalization
        # return torch.stft(x, **kwargs)
        return log_amp, pha, com
    
    def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):
        hann_window = torch.hann_window(win_size).to(audio.device)
        stft_spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                            center=center, pad_mode='reflect', normalized=False, return_complex=True)
        log_amp = torch.log(torch.abs(stft_spec)+1e-4)
        pha = torch.angle(stft_spec)

        com = torch.stack((torch.exp(log_amp)*torch.cos(pha), 
                        torch.exp(log_amp)*torch.sin(pha)), dim=-1)

        return log_amp, pha, com

    def _compute_istft(self, log_amp, pha, n_fft, hop_length, win_length, center=True):
        amp = torch.exp(log_amp)
        com = torch.complex(amp*torch.cos(pha), amp*torch.sin(pha))
        hann_window = torch.hann_window(win_length).to(com.device)
        audio = torch.istft(com, n_fft, hop_length=hop_length, win_length=win_length, window=hann_window, center=center)
        return audio
        
    def _extract_high_subbands(self, spec, subband_num=16):
        """
        Extract the highest N subbands from a spectrogram.
        
        Args:
            spec (Tensor): Input spectrogram of shape [C, F, T]
            subband_num (int): Number of highest subbands to extract (1~32)

        Returns:
            Tensor: Extracted subbands of shape [C, new_F, T]
        """
        C, F, T = spec.shape
        total_subbands = 32
        bins_per_band = F // total_subbands  # e.g. 1024 // 32 = 32
        assert 1 <= subband_num <= total_subbands, "subband_num must be between 1 and 32"

        # Compute freq bin range for high subbands
        f_start = 1 + bins_per_band * (total_subbands - subband_num)
        f_end = F  # always end at Nyquist (full)

        # pdb.set_trace()
        subbands = spec[:, f_start:f_end, :]

        # Include DC only when using all 32 subbands
        if subband_num == 32:
            dc = spec[:, 0:1, :]
            subbands = torch.cat([dc, subbands], dim=1)

        return subbands  # [C, new_F, T]
    
    def _build_encoder(self, cfg):
        # model_class = getattr(models, cfg['type'])
        # model_class(**cfg['args])
        encoder = ResnetEncoder(**cfg)
        return encoder
    
    def _build_decoder(self, cfg):
        return APNet_BWE_Model(cfg)
    
    def _build_quantizer(self, cfg):
        return ResidualVectorQuantize(**cfg)
    
    def extract_sideinfo(self, wb_input):
        # stft method
        log_amp, pha, com = self._compute_stft(wb_input)    # [B,1,L] -> [B,F,T]
        # pdb.set_trace()
        log_amp = self._extract_high_subbands(log_amp, subband_num=self.h.encoder_cfg.subband_num)
        sideinfo = self.encoder(log_amp.unsqueeze(1))       # [B,1,F,T] -> [B,D,F/S,T]
        
        # must return [B,C,T]
        return sideinfo
    
    def quantize_sideinfo(self, sideinfo, **kwargs):
        if sideinfo.dim() == 4:
            NotImplementedError      
        else:
            out = self.quantizer(sideinfo, **kwargs)
        return out
    
    def decode(self, nb_input, condition, **kwargs):
        self.decoder_type = 'freq'      # temp
        if self.decoder_type == 'freq': # frequency
            ## note: use Center=True for analysis and synthesis
            log_amp, pha, com = self._compute_stft(nb_input, pad=False, **self.h.decoder_cfg.stft)
            log_amp, pha, com = self.decoder(log_amp, pha)
            # reconstruct waveform
            out = self._compute_istft(log_amp, pha, **self.h.decoder_cfg.stft)
        else:
            NotImplementedError("Time domain input")
        return out
    
    def inference(self, nb_input, quantized_condition, **kwargs):
        NotImplementedError("Inference method not implemened.")
        
    def forward(self, nb_input, wb_input, **kwargs):
        sideinfo = self.extract_sideinfo(wb_input) # [B,1,T] -> [B,C,T]
        parameters, code, latents, commit_loss, codebook_loss = self.quantize_sideinfo(sideinfo, **kwargs) # [B,C,T]
        out = self.decode(nb_input, parameters, **kwargs) 
        return out, commit_loss, codebook_loss
    
def main():
    config = load_config("configs/config_resnet_apbwe.yaml")
    config.generator.decoder_cfg.ConvNeXt_layers = 2
    print(config.generator.encoder_cfg)
    print(config.generator.decoder_cfg)

    model = ResNet_APBWE(config)
    nb = torch.rand(5,1,48000)
    wb = torch.rand(5,1,48000)
    input_m = torch.randn(1,1,513,200)
    input_p = torch.randn(1,1,513,200)
    encoder_cfg = config.generator.encoder_cfg
    stft_input = torch.randn(5,1,32*encoder_cfg.subband_num,200)
    summary(model.encoder, input_data=[stft_input], 
            col_names=["input_size", "output_size", "num_params",], depth=3)   
    summary(model.decoder, input_data=[input_m.squeeze(1),input_p.squeeze(1)], 
            col_names=["input_size", "output_size", "num_params",], depth=3)   
    summary(model, input_data=[nb, wb], 
            col_names=["input_size", "output_size", "num_params",], depth=3)   
    
    
if __name__ == "__main__":
    main()
