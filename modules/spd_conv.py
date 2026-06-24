import torch
import torch.nn as nn

class SPD(nn.Module):
    """Space-to-Depth layer — rearranges spatial blocks into channels.
    Replaces strided convolution/pooling without information loss.
    Input:  (B, C, H, W)
    Output: (B, C*scale^2, H/scale, W/scale)
    """
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        s = self.scale
        b, c, h, w = x.shape
        # Rearrange spatial blocks into channel dimension
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * s * s, h // s, w // s)
        return x


class SPDConv(nn.Module):
    """SPD-Conv: Space-to-Depth followed by non-strided convolution.
    Replaces strided Conv(stride=2) with lossless downsampling.
    c1: input channels
    c2: output channels
    scale: spatial downsampling factor (default 2)
    """
    def __init__(self, c1, c2, scale=2, *args, **kwargs):
        super().__init__()
        self.spd = SPD(scale)
        # After SPD, channels multiply by scale^2
        self.conv = nn.Sequential(
            nn.Conv2d(c1 * scale * scale, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(self.spd(x))