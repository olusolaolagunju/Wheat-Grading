import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building Blocks ───────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Two 3×3 convs with residual connection.
    Paper: "each residual unit comprises two 3×3 convolutions"
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class AFPNUp(nn.Module):
    """AFPN-style upsampling: 1×1 conv (channel alignment) + bilinear upsample.
    Paper: "we utilize 1×1 convolution and bilinear interpolation for upsampling"

    Args:
        c1    : input channels
        c2    : output channels
        scale : upsample factor (default 2)
    """
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.scale = scale

    def forward(self, x):
        return F.interpolate(self.proj(x), scale_factor=self.scale,
                             mode='bilinear', align_corners=False)


class AFPNDown(nn.Module):
    """AFPN-style downsampling: k×k conv with stride k.
    Paper: "2×2 convolution with stride 2 for 2× downsampling"

    Args:
        c1    : input channels
        c2    : output channels
        scale : downsampling factor (default 2)
    """
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, scale, stride=scale, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ASF2(nn.Module):
    """Adaptive Spatial Fusion for 2 inputs.
    Paper sec. III-A: resolves per-location feature conflicts between levels.

    FIX vs original: inputs are projected to c2 FIRST via 1×1 convs before
    fusion. This handles the case where the two inputs have DIFFERENT channel
    counts (e.g. AFPNUp output at 64ch fused with backbone P3 at 128ch).
    Without projection, the cat would be 192ch but attn expects 128ch → crash.

    Mechanism:
        1. Project both inputs to c2 channels (handles any input channel count)
        2. Concatenate projected features → 2*c2 channels
        3. Generate 2 spatial weight maps via small conv → softmax
        4. Weighted sum: each location picks how much of each input to use

    Args:
        c1a : channels of first input  (from parse_model: ch[f[0]])
        c1b : channels of second input (from parse_model: ch[f[1]])
        c2  : output channels
    """
    def __init__(self, c1a, c1b, c2):
        super().__init__()
        # Project each input to c2 — Identity if already c2, conv otherwise
        self.proj_a = nn.Sequential(
            nn.Conv2d(c1a, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        ) if c1a != c2 else nn.Identity()

        self.proj_b = nn.Sequential(
            nn.Conv2d(c1b, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        ) if c1b != c2 else nn.Identity()

        # Attention: 2*c2 → 2 spatial weight maps
        self.attn = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, 2, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.proj_a(x[0])                        # (B, c2, H, W)
        x2 = self.proj_b(x[1])                        # (B, c2, H, W)
        w  = self.softmax(self.attn(torch.cat([x1, x2], dim=1)))  # (B, 2, H, W)
        return w[:, 0:1] * x1 + w[:, 1:2] * x2        # (B, c2, H, W)