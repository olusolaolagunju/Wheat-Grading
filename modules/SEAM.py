import torch
import torch.nn as nn
import torch.nn.functional as F

class CSMM(nn.Module):
    """Channel-Spatial Mixed Module"""
    def __init__(self, in_channels, patch_size):
        super(CSMM, self).__init__()
        # Patch embedding (stride=1 to preserve resolution)
        self.patch_embed = nn.Conv2d(in_channels, in_channels, kernel_size=patch_size, stride=1, padding=patch_size//2)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.gelu1(x)
        x = self.bn1(x)

        x = self.depthwise(x)
        x = self.gelu2(x)
        x = self.bn2(x)

        x = self.pointwise(x)
        x = self.gelu3(x)
        x = self.bn3(x)

        return x


class SEAM(nn.Module):
    def __init__(self, c1, c2, patch_sizes=[6, 7, 8]):
        super(SEAM, self).__init__()
        if isinstance(patch_sizes, int):
            patch_sizes = [patch_sizes]

        self.csmm_blocks = nn.ModuleList([CSMM(c1, p) for p in patch_sizes])

        # Fusion + channel alignment
        self.fusion = nn.AdaptiveAvgPool2d((1, 1))
        self.expand = nn.Conv2d(c1, c2, kernel_size=1)
        self.bn_expand = nn.BatchNorm2d(c2)
        self.act_expand = nn.GELU()
        
    def forward(self, x):
        # Run CSMM blocks at multiple patch sizes
        features = [csmm(x) for csmm in self.csmm_blocks]

        fused = sum(features) / len(features)  # average fusion, keeps HxW

        expanded = self.expand(fused)
        expanded = self.bn_expand(expanded)
        expanded = self.act_expand(expanded)

        return expanded
    

    