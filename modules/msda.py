import torch
import torch.nn as nn

class MSDA(nn.Module):
    """Multi-Scale Dilated Attention — lightweight version.
    Parallel dilated depthwise convolutions, averaged not concatenated.
    """
    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1

        # Parallel dilated depthwise convs — no channel expansion
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=r, dilation=r,
                          groups=c1, bias=False),  # depthwise only
                nn.BatchNorm2d(c1),
                nn.GELU()
            ) for r in [1, 2, 3, 5]
        ])

        # Single pointwise projection — no intermediate expansion
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )

        self.shortcut = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # Average multi-scale features — no concatenation
        features = [conv(x) for conv in self.dilated_convs]
        fused = sum(features) / len(features)   # same as SEAM fusion strategy
        return self.proj(fused) + self.shortcut(x)