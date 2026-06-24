import math
import torch
import torch.nn as nn


# ── Utilities ─────────────────────────────────────────────────────────────────

def autopad(k, p=None):
    """Auto-pad so output spatial size == input spatial size."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ── Base Conv (YOLO11-compatible: BN + SiLU) ──────────────────────────────────

class Conv(nn.Module):
    """Standard Conv-BN-SiLU block — matches Ultralytics YOLO11 Conv."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise Conv — groups = gcd(c1, c2)."""
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


# ── GSConv ────────────────────────────────────────────────────────────────────

class GSConv(nn.Module):
    """GSConv: Gate-Shuffle Conv.
    Li et al., "Slim-neck by GSConv", arXiv 2206.02424

    Structure:
        input (c1)
          └─ cv1: SC  → c2/2 channels   (standard conv, channel-dense)
              ├─ left half  ─────────────────────────────┐
              └─ cv2: DSC → c2/2 channels (depth-wise)  │
                                                          concat → shuffle → output (c2)

    Key idea: SC features carry full cross-channel information.
    DSC features are cheap but channel-sparse.
    Shuffling blends SC context into every DSC output channel.

    Args:
        c1  : input channels
        c2  : output channels  (must be even)
        k   : kernel size of the SC branch (default 1)
        s   : stride (default 1)
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)          # SC branch  → c_
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)          # DSC branch → c_ (groups=c_)

    def forward(self, x):
        x1 = self.cv1(x)                                      # (B, c_, H, W)
        x2 = torch.cat((x1, self.cv2(x1)), dim=1)            # (B, c2, H, W)

        # Channel shuffle — blends SC and DSC features uniformly
        # reshape → transpose → reshape back
        b, n, h, w = x2.shape
        x2 = x2.reshape(b * n // 2, 2, h * w)
        x2 = x2.permute(1, 0, 2)
        x2 = x2.reshape(2, -1, n // 2, h, w)
        return torch.cat((x2[0], x2[1]), dim=1)              # (B, c2, H, W)


# ── GSBottleneck ───────────────────────────────────────────────────────────────

class GSBottleneck(nn.Module):
    """GSBottleneck: residual block built with GSConv.

    Two parallel paths:
      conv_lighting : 1x1 GSConv → 1x1 GSConv  (fast, small receptive field)
      shortcut      : 3x3 Conv                  (skip connection)

    The 'lighting' path uses 1x1 kernels for speed.
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = Conv(c1, c2, 3, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class GSBottleneckC(GSBottleneck):
    """GSBottleneckC: variant with DWConv shortcut (lighter skip connection)."""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, 3, 1, act=False)


# ── VoVGSCSP ──────────────────────────────────────────────────────────────────

class VoVGSCSP(nn.Module):
    """VoVGSCSP: CSP block built on GSBottleneck.
    Replaces C3k2 / C2f in the YOLO neck.

    Structure (CSP-style):
        input
          ├─ cv1 → GSBottleneck → x1
          └─ cv2 ──────────────→ y
          cat(y, x1) → cv3 → output

    The VoV (One-Shot Aggregation) style means features from both
    branches are concatenated once — no iterative stacking — keeping
    the gradient path short and inference fast.

    Args:
        c1        : input channels
        c2        : output channels
        n         : number of GSBottleneck repeats (default 1)
        shortcut  : unused, kept for API compatibility with C3k2
        e         : channel expansion ratio (default 0.5)
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = GSBottleneck(c_, c_, 1, 1)
        self.res = Conv(c_, c_, 3, 1, act=False)   # residual refinement (unused in forward but kept)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))   # GSBottleneck branch
        y  = self.cv2(x)              # identity branch
        return self.cv3(torch.cat((y, x1), dim=1))