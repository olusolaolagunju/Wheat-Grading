import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module — the 'what' branch of CBAM.
    Paper eq(2): Mc(F) = sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))

    Uses BOTH average and max pooling to capture different statistics:
    - AvgPool: captures soft distributional information
    - MaxPool: captures the most distinctive features

    Both pass through a SHARED MLP with one hidden layer (reduction ratio r).
    Outputs are summed then sigmoid-activated → channel weights in [0,1].

    Args:
        channels  : number of input channels C
        reduction : bottleneck ratio r (default 16, paper default)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Shared MLP: C → C/r → C
        # Use Conv2d(1x1) instead of Linear — works on any spatial size
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # Both pooling outputs pass through the SAME MLP weights
        avg_out = self.mlp(self.avg_pool(x))   # (B, C, 1, 1)
        max_out = self.mlp(self.max_pool(x))   # (B, C, 1, 1)
        # Element-wise sum then sigmoid — paper eq(2)
        return self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial Attention Module — the 'where' branch of CBAM.
    Paper eq(3): Ms(F) = sigmoid(conv([AvgPool_c(F); MaxPool_c(F)]))

    Pools along the CHANNEL axis (not spatial) to generate two maps,
    concatenates them, then applies a single 7×7 conv → spatial weights.

    The 7×7 kernel is the paper's default (k=7 outperforms k=3 in ablation).

    Args:
        kernel_size : conv kernel for spatial attention (default 7, paper default)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        # 2 → 1 channel: two pooled maps → one spatial attention map
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool along channel dimension → two (B, 1, H, W) maps
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out = torch.max(x,  dim=1, keepdim=True)[0] # (B, 1, H, W)
        # Concatenate along channel dim → (B, 2, H, W)
        cat = torch.cat([avg_out, max_out], dim=1)
        # Conv → sigmoid → spatial weights in [0,1]
        return self.sigmoid(self.conv(cat))             # (B, 1, H, W)


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    Woo et al., ECCV 2018. https://arxiv.org/abs/1807.06521

    Sequentially applies channel then spatial attention:
        F'  = Mc(F)  ⊗ F     (channel recalibration — 'what')
        F'' = Ms(F') ⊗ F'    (spatial recalibration  — 'where')

    Channel attention: tells the network which feature maps matter
    Spatial attention: tells the network where in those maps to focus

    Paper: "we sequentially apply channel and spatial attention modules
    so that each branch can learn 'what' and 'where' to attend"

    Args:
        channels        : input (and output) channel count — CBAM is channel-preserving
        reduction       : channel attention MLP bottleneck ratio (default 16)
        spatial_kernel  : spatial attention conv kernel size (default 7)
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # Step 1: channel attention — scale feature maps by importance
        x = x * self.channel_att(x)   # (B, C, H, W) * (B, C, 1, 1) → broadcast
        # Step 2: spatial attention — scale spatial locations by importance
        x = x * self.spatial_att(x)   # (B, C, H, W) * (B, 1, H, W) → broadcast
        return x