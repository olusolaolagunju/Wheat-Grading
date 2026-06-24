import torch
import torch.nn as nn

# -----------------------------
# Star Operation (element-wise multiplication)
# -----------------------------
class StarOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x * y   # element-wise multiplication
        

class StarBlock(nn.Module):
    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.conv1 = nn.Conv2d(c1, c2, 1, 1)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1)
        self.star = StarOp()

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(a)
        return self.star(a, b)