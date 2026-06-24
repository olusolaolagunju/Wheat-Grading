import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self, c1, c2=None, k_size=None):
        super(ECA, self).__init__()
        # Adaptive kernel size based on channel count
        if k_size is None:
            t = int(abs((math.log(c1, 2) + 1) / 2))
            k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)