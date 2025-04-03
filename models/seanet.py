import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import weight_norm

class SEANet(nn.Module):
    def __init__(self, min_dim=32, strides=[2,4,8,8], 
                 c_in=1, c_out=1, out_bias=True,
                 **kwargs):
        super().__init__()
        
        self.min_dim = min_dim # first conv output channels
        self.downsampling_factor = np.prod(strides)
        self._initialize_weights()
        
        self.conv_in = Conv1d(
            in_channels=c_in,
            out_channels=min_dim,
            kernel_size=7,
            stride=1
        )
        
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, strides[0]),
                                    EncBlock(min_dim*4, strides[1]),
                                    EncBlock(min_dim*8, strides[2]),
                                    EncBlock(min_dim*16, strides[3])                                        
                                    ])
        
        self.conv_bottle = nn.Sequential(
                                        Conv1d(
                                            in_channels=min_dim*16,
                                            out_channels = min_dim*16//4,
                                            kernel_size = 7, 
                                            stride = 1,
                                            ),
                                        
                                        Conv1d(
                                            in_channels=min_dim*16//4,
                                            out_channels = min_dim*16,
                                            kernel_size = 7,
                                            stride = 1,
                                            ),
                                        )
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*8, strides[3]),
                                    DecBlock(min_dim*4, strides[2]),
                                    DecBlock(min_dim*2, strides[1]),
                                    DecBlock(min_dim, strides[0]),
                                    ])
        
        self.conv_out = Conv1d(
            in_channels=min_dim,
            out_channels=c_out,
            kernel_size=7,
            stride=1,
            bias=out_bias,
        )
        
    def _adjust_signal_len(self, x):
        if x.dim() == 3:
            B,C,sig_len = x.size() # [B,C,T]
        elif x.dim() == 2:
            B, sig_len = x.size()
        
        pad_len = sig_len % self.downsampling_factor
        if pad_len != 0:
            x = x[..., :-pad_len]
        return x, pad_len
    
    def forward(self, x):
        x, pad_len = self._adjust_signal_len(x) # adjust length
        
        # conv in
        skip = []
        x = self.conv_in(x)
        skip.append(x)
        # enc
        for encoder in self.encoder:
            x = encoder(x)
            skip.append(x)
        # bottleneck
        x = self.conv_bottle(x)
        # dec
        skip = skip[::-1]
        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)
        x = x + skip[4]
        # conv out
        x = self.conv_out(x)
        
        # pad
        padval = torch.zeros([x.size(0), x.size(1), pad_len]).to(x.device) # [B,C_out,T]
        x = torch.cat((x, padval), dim=-1)
        
        return x
    
    def _initialize_weights(self):
        # Iterate through all layers and apply Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    

class EncBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()
        

        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels//2, 1),
                                    ResUnit(out_channels//2, 3),
                                    ResUnit(out_channels//2, 9)                                        
                                    ])
        
        self.conv = nn.Sequential(
                    nn.ELU(),
                    Pad((2 * stride - 1, 0)),
                    nn.Conv1d(in_channels = out_channels//2,
                                       out_channels = out_channels,
                                       kernel_size = 2 * stride,
                                       stride = stride, padding = 0),
                    )  
        
    def forward(self, x):
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.conv(x)
        return x
    
class DecBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        ## Upsampling
        self.conv = ConvTransposed1d(
                                 in_channels = out_channels*2, 
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                 )
        
        
        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels, 1),
                                    ResUnit(out_channels, 3),
                                    ResUnit(out_channels, 9)                                       
                                    ])
               
        self.stride = stride
        

    def forward(self, x):
        x = self.conv(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x
    
class ResUnit(nn.Module):
    def __init__(self, channels, dilation = 1):
        super().__init__()
        

        self.conv_in = Conv1d(
                                 in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 3, stride= 1,
                                 dilation = dilation,
                                 )
        
        self.conv_out = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        
        self.conv_shortcuts = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )

    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_out(y)
        x = self.conv_shortcuts(x)
        return x + y

class Conv1d(nn.Module):
    """ Causal Conv1d """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels,
            kernel_size= kernel_size, 
            stride= stride, 
            dilation = dilation,
            groups = groups,
            bias=bias,
        )
        self.conv = weight_norm(self.conv)
        self.pad = Pad(((kernel_size-1)*dilation, 0)) 
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

class ConvTransposed1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride =stride,
            dilation = dilation,
            bias=bias,
        )
        self.conv = weight_norm(self.conv)
        self.pad = dilation * (kernel_size - 1) - dilation * (stride - 1)
        self.activation = nn.ELU()
        
    def forward(self, x):
        x = self.conv(x)
        x = x[..., :-self.pad]
        x = self.activation(x)
        return x
    
class Pad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        return F.pad(x, pad=self.pad)    

if __name__ == "__main__":
    """ SEANet """
    from torchinfo import summary
    model = SEANet(c_out=1, c_in=1, min_dim=56, out_bias=True)
    wav = torch.rand(4,1,55400)
    summary(
        model, input_data = wav,
        col_names=['input_size','output_size'],
        depth=2
    )