import torch
import torch.nn as nn

class SRM(nn.Module):
    """Spatial Refinement Module"""
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        r = self.dw(x)
        r = self.pw(r)
        r = self.bn(r)
        return x + self.act(r)
