import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import numpy as np
from typing import Tuple, List

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value= None,
        adanorm_num_embeddings = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim*3)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim*3, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id = None) :
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class APNet_BWE_Model(torch.nn.Module):
    def __init__(self, h, **kwargs):
        super(APNet_BWE_Model, self).__init__()
        self.h = h
        self.adanorm_num_embeddings = None
        layer_scale_init_value =  1 / h.ConvNeXt_layers

        self.conv_pre_mag = nn.Conv1d(h.n_fft//2+1, h.ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_mag = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(h.n_fft//2+1, h.ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_pha = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)

        self.convnext_mag = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=h.ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(h.ConvNeXt_layers)
            ]
        )

        self.convnext_pha = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=h.ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(h.ConvNeXt_layers)
            ]
        )

        self.norm_post_mag = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.apply(self._init_weights)
        self.linear_post_mag = nn.Linear(h.ConvNeXt_channels, h.n_fft//2+1)
        self.linear_post_pha_r = nn.Linear(h.ConvNeXt_channels, h.n_fft//2+1)
        self.linear_post_pha_i = nn.Linear(h.ConvNeXt_channels, h.n_fft//2+1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mag_nb, pha_nb, cond_mag):

        x_mag = self.conv_pre_mag(mag_nb)
        x_pha = self.conv_pre_pha(pha_nb)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)

        for conv_block_mag, conv_block_pha in zip(self.convnext_mag, self.convnext_pha):
            x_mag = x_mag + x_pha + cond_mag
            x_pha = x_pha + x_mag
            x_mag = conv_block_mag(x_mag, cond_embedding_id=None)
            x_pha = conv_block_pha(x_pha, cond_embedding_id=None) 

        x_mag = self.norm_post_mag(x_mag.transpose(1, 2))
        mag_wb = mag_nb + self.linear_post_mag(x_mag).transpose(1, 2)

        x_pha = self.norm_post_pha(x_pha.transpose(1, 2))
        x_pha_r = self.linear_post_pha_r(x_pha)
        x_pha_i = self.linear_post_pha_i(x_pha)
        pha_wb = torch.atan2(x_pha_i, x_pha_r).transpose(1, 2)

        com_wb = torch.stack((torch.exp(mag_wb)*torch.cos(pha_wb), 
                           torch.exp(mag_wb)*torch.sin(pha_wb)), dim=-1)
        
        return mag_wb, pha_wb, com_wb

def stft_mag(audio, n_fft=2048, hop_length=512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
    stft_mag = torch.abs(stft_spec)
    return(stft_mag)

def cal_snr(pred, target):
    snr = (20 * torch.log10(torch.norm(target, dim=-1) / torch.norm(pred - target, dim=-1).clamp(min=1e-8))).mean()
    return snr

def cal_lsd(pred, target):
    sp = torch.log10(stft_mag(pred).square().clamp(1e-8))
    st = torch.log10(stft_mag(target).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()