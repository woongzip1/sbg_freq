from models.model import ConvNeXtBlock
import torch
import torch.nn as nn
import pdb

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class VocosBWE(torch.nn.Module):
    def __init__(self, h, **kwargs):
        super(VocosBWE, self).__init__()
        self.h = h
        self.adanorm_num_embeddings = None
        layer_scale_init_value =  1 / h.ConvNeXt_layers

        self.conv_pre_mag = nn.Conv1d(h.n_fft//2+1, h.ConvNeXt_channels//2, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_mag = nn.LayerNorm(h.ConvNeXt_channels//2, eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(h.n_fft//2+1, h.ConvNeXt_channels//2, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_pha = nn.LayerNorm(h.ConvNeXt_channels//2, eps=1e-6)

        self.convnext_blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=h.ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(h.ConvNeXt_layers)
            ]
        )

        self.norm_post = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.linear_out = nn.Linear(h.ConvNeXt_channels, (h.n_fft + 2))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mag_nb, pha_nb, cond_mag):

        x_mag = self.conv_pre_mag(mag_nb)
        x_pha = self.conv_pre_pha(pha_nb)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)

        x = torch.cat([x_mag, x_pha], dim=1)
        for conv_block in self.convnext_blocks:
            # pdb.set_trace()
            x = x + cond_mag
            x = conv_block(x, cond_embedding_id=None)

        x = self.norm_post(x.transpose(1, 2)) # [B,T,C]
        x = self.linear_out(x).transpose(1,2) # [B,T,NFFT+2] -> [B,NFFT+2,T]

        ## ISTFT head 
        mag_out, pha_wb = x.chunk(2, dim=1)
        mag_wb = mag_nb + mag_out # residual (can be erased)
        com_wb = torch.stack((torch.exp(mag_wb)*torch.cos(pha_wb), 
                           torch.exp(mag_wb)*torch.sin(pha_wb)), dim=-1)
        ## Vocos
        # mag_wb = torch.exp(mag_wb)
        # mag_wb = torch.clip(mag_wb, max=1e2)
        # x = torch.cos(pha_wb)
        # y = torch.sin(pha_wb)
        # pha_wb = x + 1j * y
        # com_wb = mag_wb * pha_wb
        ## pha_wb = torch.atan2(y, x)
        ## audio = torch.istft(com_wb)
        ##
        return mag_wb, pha_wb, com_wb


def main():
    from box import Box
    from torchinfo import summary

    config = {
        "ConvNeXt_channels": 768,
        "ConvNeXt_layers": 8,
        "n_fft": 1024,
    }

    config = Box(config)
    print(config)
    model = VocosBWE(config)
    print(model)

    x = torch.randn(1,513,256) # nfft 1024 spectrum
    cond = torch.randn(1,config.ConvNeXt_channels,256)
    out = model(x,x,cond)
    print(out[0].shape)

    print(summary(model, input_data=[x,x,cond]))
    return

if __name__ == "__main__":
    main()