import torch
import torch.nn as nn

class ShuffleV2Block(nn.Module):
    def __init__(self, c1, c2, stride=1, *args, **kwargs):
        super().__init__()
        self.stride = stride
        mid = c2 // 2

        if stride == 1:
            # Basic unit — channel split
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
                nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
            )
        else:
            # Downsampling unit — no channel split
            self.branch1 = nn.Sequential(
                nn.Conv2d(c1, c1, 3, stride, 1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(c1, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
                nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.SiLU(),
            )

    @staticmethod
    def channel_shuffle(x, groups=2):
        b, c, h, w = x.shape
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self.channel_shuffle(out)