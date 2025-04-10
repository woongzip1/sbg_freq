import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.nn.utils.weight_norm as weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torchinfo import summary
from einops import rearrange
"""
ResNet18 based 2D STFT Encoder.
Adopts caual convolutions and weight normalizations
[B,1,F,T] -> [B,8D,F/32,T]; D:conv_dim
"""

class ResnetEncoder(nn.Module):
    def __init__(self, input_ch=1, conv_dim=64, subband_num=10, visualize=False, output_dim=512, **kwargs):
        super().__init__()       
        self.resnet = ResNet18(input_ch=input_ch, conv_dim=conv_dim, visualize=visualize)  # outputs: [B, D, F/s, T]
        # self.projector = SimpleLinear(subband_num=subband_num, dim_per_freqbins=conv_dim*8)  # expects [B, T, D*F/s]
        self.projector = BandProjection(in_channels=conv_dim*8, out_channels=32, subband_num=subband_num, out_freq_bins=output_dim)
    def forward(self, x):
        """
        input:  [B,1,F,T]
        output: [B,C,T]
        """
        x = self.resnet(x)      # [B,1,F,T]->[B,D,F/32,T]
        ## rearrange
        # x = rearrange(x, 'b d f t -> b t (d f)') # [B,T,D*F/32]       
        x = self.projector(x)   # [B,T,subband_num*32]
        x = rearrange(x, 'b t c -> b c t')        
        return x

class BandProjection(nn.Module):
    """
    1) 1x1 Conv로 채널을 줄임 (ex: 512 -> 16)
    2) [out_channels * subband_num]을 flatten 후
    3) Linear 로 513차원으로 projection
    
    ## out_channels 16으로 해도 되겠는데 NFFT 2048이면, subband 32면..?
    """
    
    def __init__(self, in_channels=512, out_channels=16, subband_num=17, out_freq_bins=513):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.subband_num = subband_num  # F축 = subband 개수
        self.out_freq_bins = out_freq_bins

        # 1x1 Conv: 채널만 줄이기
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Flatten 후 projection to 513
        in_features = out_channels * subband_num
        self.linear = nn.Linear(in_features, out_freq_bins)

    def forward(self, x):
        """
        Args:
            x: [B, C, F, T]  (ex: [5, 512, 17, 200])
        Returns:
            out: [B, T, 513]
        """
        B, C, F, T = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert F == self.subband_num, f"Expected F={self.subband_num}, got {F}"

        # Step 1: 1x1 Conv to reduce channels → [B, out_channels, F, T]
        x = self.conv_1x1(x)

        # Step 2: reshape to [B, T, out_channels * F]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, out_channels, F]
        x = x.view(B, T, self.out_channels * self.subband_num)  # [B, T, in_features]

        # Step 3: Linear projection to 513
        out = self.linear(x)  # [B, T, 513]

        return out

class SimpleLinear(nn.Module):
    """Very Simple Linear Module: (subband_num * dim_per_freqbins) -> 513"""
    def __init__(self, subband_num=10, dim_per_freqbins=512):
        super().__init__()
        self.subband_num = subband_num
        self.dim_per_freqbins = dim_per_freqbins
        self.in_dim = subband_num * dim_per_freqbins
        self.out_dim = 513
        self.linear = nn.Linear(self.in_dim, self.out_dim)  # naive: 전체 -> 513

    def forward(self, embeddings):
        """
        Args:
            embeddings: shape [B, T, subband_num * dim_per_freqbins]
        Returns:
            shape [B, T, 513]
        """
        B, T, D = embeddings.shape
        assert D == self.in_dim, \
            f"Expected input dim {self.in_dim}, but got {D}"

        out = self.linear(embeddings)
        return out

class BandwiseLinear(nn.Module):
    def __init__(self, subband_num=10, dim_per_freqbins=512):
        super().__init__()
        self.subband_num = subband_num
        self.dim_per_freqbins = dim_per_freqbins #  8D for feature encoder
        self.layers = nn.ModuleList([nn.Linear(self.dim_per_freqbins,32) for _ in range(self.subband_num)])
    
    def forward(self, embeddings):
        assert embeddings.shape[-1] == self.subband_num * self.dim_per_freqbins, \
            f"Expected input dim {self.subband_num * self.dim_per_freqbins}, but got {embeddings.shape[-1]}"
            
        outs = []
        for idx, layer in enumerate(self.layers):
            patch_embeddings = embeddings[:, :, idx*self.dim_per_freqbins:(idx+1)*self.dim_per_freqbins] # project per subband
            out = layer(patch_embeddings)
            outs.append(out)
        final_output = torch.cat(outs, dim=-1)

        return final_output

def ResNet18(conv_dim=64, input_ch=1, visualize=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_ch, conv_dim, visualize=visualize)

class ResNet(nn.Module):
    def __init__(self, block, layers, input_ch=1, conv_dim=64, visualize=False):
        super(ResNet, self).__init__()
        self.conv_dim = conv_dim
        self.first_conv_dim = conv_dim
        
        self.conv1 = weight_norm(CausalConv2d(input_ch, self.first_conv_dim, kernel_size=(7,7), stride=(2,1), bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2,1), padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, self.first_conv_dim, layers[0])
        self.layer2 = self._make_layer(block, self.first_conv_dim*2, layers[1], stride=(2,1))
        self.layer3 = self._make_layer(block, self.first_conv_dim*4, layers[2], stride=(2,1))
        self.layer4 = self._make_layer(block, self.first_conv_dim*8, layers[3], stride=(2,1), isfinal=True)

    def _make_layer(self, block, out_channels, blocks, stride=1, isfinal=False):
        downsample = None
        if stride != 1 : # Downsampling layer needs channel modification
            downsample = CausalConv2d(self.conv_dim, out_channels,
                             kernel_size=1, stride=stride, bias=False)
                        
        layers = []
        layers.append(block(self.conv_dim, out_channels, stride, downsample, isfinal=isfinal))
        self.conv_dim = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.conv_dim, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 1, F, T) - Input spectrogram
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # ResNet Layers with optional FiLM
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        for layer in layers:
            x = layer(x)  # Apply ResNet Layer
        return x

class BasicBlock(nn.Module):
    def __init__(self, conv_dim, out_channels, stride=1, downsample=None, isfinal=False):
        super(BasicBlock, self).__init__()
        self.conv1 = weight_norm(CausalConv2d(conv_dim, out_channels, kernel_size=3, stride=stride, bias=False))
        # self.conv1 = nn.Conv2d(conv_dim, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 =  weight_norm(CausalConv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False))
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample  # Used for downsampling (channel modificiation)
        self.isfinal = isfinal

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        # Skip ReLU for Final output
        if not self.isfinal:
            out = self.relu(out)

        return out    

""" Causal Conv """
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Calculate padding for temporal dimension (T)
        self.temporal_padding = self.dilation[1] * (self.kernel_size[1] - 1)

        # Calculate total padding for frequency dimension (F)
        total_f_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        self.frequency_padding_top = total_f_padding // 2
        self.frequency_padding_bottom = total_f_padding - self.frequency_padding_top
        
    def forward(self, x):
        ## Apply padding: F (top and bottom), T (only to the left)
        # print(f"Temporal Padding (T): {self.temporal_padding}")
        # print(f"Frequency Padding (F): top={self.frequency_padding_top}, bottom={self.frequency_padding_bottom}")
        x = F.pad(x, [self.temporal_padding, 0, self.frequency_padding_top, self.frequency_padding_bottom])
        return self._conv_forward(x, self.weight, self.bias)

def main():
    ## usage
    feature_map = torch.rand(1,1,512,37) # [B,C,F,T]
    model = ResNet18(conv_dim=64)
    out = model(feature_map)
    print(summary(
        model, input_data=feature_map,
        col_names=['input_size','output_size','num_params'], depth=3
        )   
    )
    print(out.shape)

if __name__ == "__main__":
    main()