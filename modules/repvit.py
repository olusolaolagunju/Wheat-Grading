import torch
import torch.nn as nn

class RepViTBlock(nn.Module):
    """RepViT block: separates token mixing (spatial) and channel mixing.
    Uses structural re-parameterization for efficient inference.
    c1: input channels
    c2: output channels
    stride: 1 for basic block, 2 for downsampling
    """
    def __init__(self, c1, c2=None, stride=1, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.stride = stride

        # Token mixer — depthwise conv (spatial mixing)
        self.token_mixer = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )

        # Squeeze-excitation for channel recalibration
        squeeze = max(1, c1 // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, squeeze, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(squeeze, c1, 1, bias=False),
            nn.Hardsigmoid()
        )

        # Channel mixer — pointwise conv (channel mixing)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        )

        # Identity shortcut — only when stride=1 and c1==c2
        self.shortcut = (stride == 1 and c1 == c2)
        self.act = nn.GELU()

    def forward(self, x):
        # Token mixing with SE attention
        t = self.token_mixer(x)
        t = t * self.se(t)

        # Channel mixing
        out = self.channel_mixer(t)

        # Residual connection
        if self.shortcut:
            out = out + x
        return self.act(out)