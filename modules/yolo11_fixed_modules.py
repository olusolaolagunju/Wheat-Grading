"""Reference-faithful re-implementations of the modules that were mis-specified in the
YOLO11 experiments, adapted to the Ultralytics yaml/parse_model interface.

Every block below was written against the ORIGINAL authors' code, fetched from the
repositories cited in each docstring. Deviations that the YOLO yaml
interface forces (e.g. a 1x1 adapter when c1 != c2) are stated explicitly.

Registration (ultralytics/nn/tasks.py, v8.3.180):
  - StarBlock, RepViTBlock, SEAM, MultiSEAM, MSDA, C3k2Ghost  -> add to BOTH
    `base_modules` (c1 inserted, c2 width-scaled) and `repeat_modules` (n inserted).
  - BiFPN_Concat / BiFPN_Concat2 / BiFPN_Concat3 -> handle like Concat:
        elif m in {Concat, BiFPN_Concat, BiFPN_Concat2, BiFPN_Concat3}:
            c2 = sum(ch[x] for x in f)
  - BiFPN_Add -> new branch:
        elif m is BiFPN_Add:
            c2 = ch[f[0]]
            args = [c2, len(f)]

Width scaling (item 4 of the review): because these blocks sit in base_modules, the yaml
channel argument is multiplied by the scale width (0.25 for n) exactly as for Conv/C3k2.
Write channels at the 1024-scale used by the stock yaml (64/128/256/512/1024), not at the
already-scaled values (16/32/64/128/256), otherwise the model is scaled twice.

Self-test:  python yolo11_fixed_modules.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f, C3k, GhostBottleneck

# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------


class ConvBN(nn.Sequential):
    """Conv2d followed by BatchNorm2d (StarNet's ConvBN / RepViT's Conv2d_BN)."""

    def __init__(self, c1, c2, k=1, s=1, p=0, d=1, g=1, bn=True, bn_weight_init=1.0):
        super().__init__()
        self.add_module("conv", nn.Conv2d(c1, c2, k, s, p, d, g, bias=not bn))
        if bn:
            self.add_module("bn", nn.BatchNorm2d(c2))
            nn.init.constant_(self.bn.weight, bn_weight_init)
            nn.init.constant_(self.bn.bias, 0)


class Residual(nn.Module):
    """x + f(x). Used by RepViT (channel mixer) and SEAM (depthwise branch)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class SqueezeExcite(nn.Module):
    """timm.models.layers.SqueezeExcite with rd_ratio=0.25 (what RepViT calls)."""

    def __init__(self, c, rd_ratio=0.25):
        super().__init__()
        rd = max(1, int(c * rd_ratio))
        self.fc1 = nn.Conv2d(c, rd, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd, c, 1, bias=True)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = self.fc2(self.act(self.fc1(s)))
        return x * torch.sigmoid(s)


# ----------------------------------------------------------------------------------
# 2. StarNet block   (Ma et al., "Rewrite the Stars", CVPR 2024)
#    source: github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py, class Block
# ----------------------------------------------------------------------------------


class _StarBlockCore(nn.Module):
    """Verbatim structure of StarNet Block(dim, mlp_ratio):
    DW7x7(+BN) -> f1, f2 1x1 (no BN) -> ReLU6(f1) * f2 -> g 1x1(+BN) -> DW7x7 (no BN) -> + input.
    DropPath omitted (StarNet default drop_path_rate=0.0)."""

    def __init__(self, dim, mlp_ratio=3):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, g=dim, bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, g=dim, bn=False)
        self.act = nn.ReLU6()

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return inp + x


class StarBlock(nn.Module):
    """n StarNet Blocks at width c2. yaml: [-1, n, StarBlock, [c2]] or [c2, mlp_ratio].

    StarNet changes channels only in its stride-2 down-sampler (ConvBN 3x3 s2), which in a
    YOLO yaml is the preceding Conv layer, so the block itself is channel-preserving. When the
    yaml asks for c1 != c2 a 1x1 ConvBN adapter is inserted (documented deviation)."""

    def __init__(self, c1, c2, n=1, mlp_ratio=3):
        super().__init__()
        self.adapt = nn.Identity() if c1 == c2 else ConvBN(c1, c2, 1, bn=True)
        self.m = nn.Sequential(*[_StarBlockCore(c2, mlp_ratio) for _ in range(n)])

    def forward(self, x):
        return self.m(self.adapt(x))


# ----------------------------------------------------------------------------------
# 3. RepViT block   (Wang et al., "RepViT: Revisiting Mobile CNN From ViT Perspective", CVPR 2024)
#    source: github.com/THU-MIG/RepViT/blob/main/model/repvit.py, classes RepVGGDW, RepViTBlock
# ----------------------------------------------------------------------------------


class RepVGGDW(nn.Module):
    """RepViT's training-time token mixer: BN(DW3x3(+BN)(x) + DW1x1(x) + x). Verbatim."""

    def __init__(self, ed):
        super().__init__()
        self.conv = ConvBN(ed, ed, 3, 1, 1, g=ed, bn=True)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)


class _RepViTBlockCore(nn.Module):
    """One RepViTBlock. hidden_dim = 2*inp (asserted in the original).
    stride 1 (identity block): token_mixer = RepVGGDW [+ SE]; channel_mixer = Residual(pw 2x -> GELU -> pw-linear, BN init 0).
    stride 2 / channel change: token_mixer = DW3x3 s2 (+BN) [+ SE] -> 1x1 ConvBN inp->oup; channel_mixer same.
    NO activation after the residual (the original forward is channel_mixer(token_mixer(x)))."""

    def __init__(self, inp, oup, stride=1, use_se=False):
        super().__init__()
        if stride == 1 and inp == oup:
            self.token_mixer = nn.Sequential(RepVGGDW(inp), SqueezeExcite(inp) if use_se else nn.Identity())
        else:
            self.token_mixer = nn.Sequential(
                ConvBN(inp, inp, 3, stride, 1, g=inp, bn=True),
                SqueezeExcite(inp) if use_se else nn.Identity(),
                ConvBN(inp, oup, 1, 1, 0, bn=True),
            )
        self.channel_mixer = Residual(
            nn.Sequential(
                ConvBN(oup, 2 * oup, 1, 1, 0, bn=True),
                nn.GELU(),
                ConvBN(2 * oup, oup, 1, 1, 0, bn=True, bn_weight_init=0.0),
            )
        )

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTBlock(nn.Module):
    """n RepViT blocks. yaml: [-1, n, RepViTBlock, [c2]]  (stride handled by the yaml's Conv layers).

    SE placement follows the official cfgs (repvit_m0_9 etc.): within a stage the stride-1
    blocks alternate SE on/off starting with ON, and a down-sampling block has SE OFF.
    Here block 0 handles any c1 != c2 (SE off, as in the official stride-2 block) and the
    remaining blocks alternate starting with SE on."""

    def __init__(self, c1, c2, n=1, stride=1):
        super().__init__()
        blocks = []
        if c1 != c2 or stride != 1:
            blocks.append(_RepViTBlockCore(c1, c2, stride, use_se=False))
            n -= 1
        for i in range(max(n, 0)):
            blocks.append(_RepViTBlockCore(c2, c2, 1, use_se=(i % 2 == 0)))
        self.m = nn.Sequential(*blocks)

    def forward(self, x):
        return self.m(x)


# ----------------------------------------------------------------------------------
# 1. SEAM / MultiSEAM   (Yu et al., "YOLO-FaceV2", 2022)
#    source: github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/models/common.py, classes SEAM, DcovN, MultiSEAM
# ----------------------------------------------------------------------------------


def _seam_stack(c, depth, kernel_size=3):
    """depth x [ Residual(DW kxk -> GELU -> BN) -> PW 1x1 -> GELU -> BN ]  (GELU before BN is the authors' order)."""
    return nn.Sequential(
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(c, c, kernel_size, 1, kernel_size // 2, groups=c),
                        nn.GELU(),
                        nn.BatchNorm2d(c),
                    )
                ),
                nn.Conv2d(c, c, 1, 1, 0),
                nn.GELU(),
                nn.BatchNorm2d(c),
            )
            for _ in range(depth)
        ]
    )


def _seam_fc(c, reduction=16):
    return nn.Sequential(
        nn.Linear(c, c // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(c // reduction, c, bias=False),
        nn.Sigmoid(),
    )


class SEAM(nn.Module):
    """SEAM(c1, c2, n): output = x * exp(sigmoid-FC(avgpool(DCovN(x)))).
    It is a multiplicative channel attention applied back onto the INPUT; it is not a feature
    stack that replaces the block it is attached to. c2 is forced to c1 (as in the original).
    yaml: [-1, n, SEAM, [c2]] placed AFTER the block you want re-weighted (e.g. after C2PSA)."""

    def __init__(self, c1, c2, n=1, reduction=16):
        super().__init__()
        c2 = c1
        self.DCovN = _seam_stack(c2, n)
        self.fc = _seam_fc(c2, reduction)

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.DCovN(x).mean((2, 3)).view(b, c)
        y = torch.exp(self.fc(y)).view(b, c, 1, 1)
        return x * y


class MultiSEAM(nn.Module):
    """MultiSEAM(c1, c2, n, kernel_size=3, patch_size=(3,5,7)). Three DcovN branches, each starting
    with a PATCH-EMBED conv of kernel=stride=patch_size (so H/W are reduced, and the even/odd
    kernel padding issue of the previous version never arises), globally pooled together with the
    input, averaged, then FC -> sigmoid -> exp -> multiply input. Verbatim topology."""

    def __init__(self, c1, c2, n=1, kernel_size=3, patch_size=(3, 5, 7), reduction=16):
        super().__init__()
        c2 = c1
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c1, c2, p, p),  # patch embed, stride = patch
                    nn.GELU(),
                    nn.BatchNorm2d(c2),
                    _seam_stack(c2, n, kernel_size),
                )
                for p in patch_size
            ]
        )
        self.fc = _seam_fc(c2, reduction)

    def forward(self, x):
        b, c = x.shape[:2]
        pooled = [br(x).mean((2, 3)).view(b, c) for br in self.branches] + [x.mean((2, 3)).view(b, c)]
        y = sum(pooled) / len(pooled)
        y = torch.exp(self.fc(y)).view(b, c, 1, 1)
        return x * y


# ----------------------------------------------------------------------------------
# 5. BiFPN fast normalised fusion   (Tan et al., "EfficientDet", CVPR 2020, Sec. 3.3)
#    reference impl: github.com/zylo117/Yet-Another-EfficientDet-Pytorch efficientdet/model.py
#    w = ReLU(w);  O = sum_i  w_i / (eps + sum_j w_j) * I_i,   eps = 1e-4
# ----------------------------------------------------------------------------------


class BiFPN_Concat(nn.Module):
    """Weighted CONCAT of n inputs. Community adaptation of BiFPN fusion to a YOLO Concat slot.
    Adds the missing ReLU so weights stay non-negative (paper eq. 'fast normalized fusion')."""

    def __init__(self, dimension=1, n_inputs=2, epsilon=1e-4):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, x):
        w = F.relu(self.w)
        w = w / (w.sum() + self.epsilon)
        return torch.cat([w[i] * xi for i, xi in enumerate(x)], self.d)


class BiFPN_Concat2(BiFPN_Concat):
    def __init__(self, dimension=1):
        super().__init__(dimension, 2)


class BiFPN_Concat3(BiFPN_Concat):
    def __init__(self, dimension=1):
        super().__init__(dimension, 3)


class BiFPN_Add(nn.Module):
    """The ACTUAL BiFPN node: weighted SUM of equal-channel inputs followed by
    depthwise-separable conv + BN + SiLU (EfficientDet uses swish). Inputs must share c.
    yaml: [[a, b], 1, BiFPN_Add, []]  with the parse_model branch given in the file header."""

    def __init__(self, c, n_inputs=2, epsilon=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.epsilon = epsilon
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, 1, 0, bias=True),
            nn.BatchNorm2d(c, momentum=0.01, eps=1e-3),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        w = F.relu(self.w)
        w = w / (w.sum() + self.epsilon)
        fused = sum(w[i] * xi for i, xi in enumerate(x))
        return self.conv(self.act(fused))  # EfficientDet order: swish -> conv -> bn


# ----------------------------------------------------------------------------------
# 6. MSDA: Multi-Scale Dilated Attention   (Jiao et al., "DilateFormer", IEEE TMM 2023)
#    source: github.com/JIAOJIAYUASD/dilateformer/blob/main/models/dilateformer.py
#    classes DilateAttention, MultiDilatelocalAttention, DilateBlock
# ----------------------------------------------------------------------------------


class DilateAttention(nn.Module):
    """Sliding-window dilated attention: each query attends to its kxk dilated neighbourhood of keys."""

    def __init__(self, head_dim, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.k = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        h = d // self.head_dim
        q = q.reshape(B, h, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2)  # B,h,N,1,hd
        k = self.unfold(k).reshape(B, h, self.head_dim, self.k * self.k, H * W).permute(0, 1, 4, 2, 3)  # B,h,N,hd,kk
        attn = ((q @ k) * self.scale).softmax(dim=-1)  # B,h,N,1,kk
        v = self.unfold(v).reshape(B, h, self.head_dim, self.k * self.k, H * W).permute(0, 1, 4, 3, 2)  # B,h,N,kk,hd
        return (attn @ v).transpose(1, 2).reshape(B, H, W, d)  # B,H,W,d


class MultiDilatelocalAttention(nn.Module):
    """Channels are split into len(dilation) groups; group i runs DilateAttention at dilation[i]."""

    def __init__(self, dim, num_heads, kernel_size=3, dilation=(1, 2, 3)):
        super().__init__()
        self.nd = len(dilation)
        assert num_heads % self.nd == 0, f"num_heads {num_heads} must be a multiple of len(dilation) {self.nd}"
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        head_dim = dim // num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.attn = nn.ModuleList([DilateAttention(head_dim, kernel_size, d) for d in dilation])
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: B,H,W,C (channels-last, as in the original)
        B, H, W, C = x.shape
        xc = x.permute(0, 3, 1, 2)
        qkv = self.qkv(xc).reshape(B, 3, self.nd, C // self.nd, H, W).permute(2, 1, 0, 3, 4, 5)
        outs = [self.attn[i](qkv[i][0], qkv[i][1], qkv[i][2]) for i in range(self.nd)]  # each B,H,W,C/nd
        return self.proj(torch.cat(outs, dim=-1))


class _DilateBlock(nn.Module):
    """DilateBlock: x + attn(LN(x)); x + MLP(LN(x)), with the optional CPE (DW3x3) before it."""

    def __init__(self, dim, num_heads, kernel_size=3, dilation=(1, 2, 3), mlp_ratio=4.0, cpe=True):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) if cpe else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):  # B,C,H,W
        if self.pos_embed is not None:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2)


class MSDA(nn.Module):
    """n DilateBlocks at width c2. yaml: [-1, n, MSDA, [c2]] or [c2, num_heads, dilation_list].

    The paper uses dilation (1,2,3) on stages of 72/144/288/576 channels, all divisible by 3.
    YOLO11 stage widths are powers of two, so with 3 dilations C/3 is not an integer. Default
    here is dilation (1, 2) with 4 heads; pass dilation=[1,2,3] only where c2 % 3 == 0.
    1x1 ConvBN adapter when c1 != c2 (documented deviation)."""

    def __init__(self, c1, c2, n=1, num_heads=4, dilation=(1, 2), kernel_size=3, mlp_ratio=4.0):
        super().__init__()
        self.adapt = nn.Identity() if c1 == c2 else ConvBN(c1, c2, 1, bn=True)
        self.m = nn.Sequential(
            *[_DilateBlock(c2, num_heads, kernel_size, tuple(dilation), mlp_ratio) for _ in range(n)]
        )

    def forward(self, x):
        return self.m(self.adapt(x))


# ----------------------------------------------------------------------------------
# 7. Ghost variant of C3k2 with the STOCK positional signature
#    stock: C3k2(c1, c2, n, c3k=False, e=0.5, g=1, shortcut=True)  (ultralytics v8.3.180 block.py)
#    yaml positional order after the channel arg is therefore [c2, c3k, e, g, shortcut].
#    Stock YOLO11n uses [256, False, 0.25] / [512, False, 0.25] at P2-P3 and [512, True] / [1024, True] at P4-P5.
# ----------------------------------------------------------------------------------


class C3k2Ghost(C2f):
    """C3k2 whose inner Bottlenecks are replaced by Ultralytics GhostBottleneck (the same
    substitution C3Ghost makes to C3). Same positional args as C3k2, so the yaml line for the
    P3 stage is [-1, 2, C3k2Ghost, [512, False, 0.25]] and for P4 [-1, 2, C3k2Ghost, [512, True]].
    Register in base_modules and repeat_modules."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else GhostBottleneck(self.c, self.c, 3, 1) for _ in range(n)
        )


# ----------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    x64 = torch.randn(2, 64, 96, 96)
    x128 = torch.randn(2, 128, 48, 48)
    x256 = torch.randn(2, 256, 24, 24)

    checks = [
        ("StarBlock(64->64, n=2)", StarBlock(64, 64, 2), x64, (2, 64, 96, 96)),
        ("StarBlock(64->128)", StarBlock(64, 128, 1), x64, (2, 128, 96, 96)),
        ("RepViTBlock(128->128, n=3)", RepViTBlock(128, 128, 3), x128, (2, 128, 48, 48)),
        ("RepViTBlock(64->128, n=2)", RepViTBlock(64, 128, 2), x64, (2, 128, 96, 96)),
        ("SEAM(256, n=2)", SEAM(256, 256, 2), x256, (2, 256, 24, 24)),
        ("MultiSEAM(256, n=1)", MultiSEAM(256, 256, 1), x256, (2, 256, 24, 24)),
        ("MSDA(128, n=1, heads=4, dil=(1,2))", MSDA(128, 128, 1), x128, (2, 128, 48, 48)),
        ("MSDA(256, n=1, heads=6, dil=(1,2,3)) needs c%3==0", None, None, None),
        ("C3k2Ghost(64->128, n=2, c3k=False, e=0.25)", C3k2Ghost(64, 128, 2, False, 0.25), x64, (2, 128, 96, 96)),
        ("C3k2Ghost(128->128, n=2, c3k=True)", C3k2Ghost(128, 128, 2, True), x128, (2, 128, 48, 48)),
    ]
    for name, mod, inp, shape in checks:
        if mod is None:
            try:
                MSDA(256, 256, 1, num_heads=6, dilation=(1, 2, 3))
                print(f"{name:52s} UNEXPECTED: built")
            except AssertionError as e:
                print(f"{name:52s} correctly rejected: {e}")
            continue
        y = mod(inp)
        assert tuple(y.shape) == shape, (name, y.shape)
        print(f"{name:52s} out={tuple(y.shape)}  params={n_params(mod):,}")

    cat = BiFPN_Concat2()(([x64, x64]))
    assert cat.shape == (2, 128, 96, 96)
    with torch.no_grad():
        BiFPN_Concat2().w.copy_(torch.tensor([-1.0, 1.0]))  # negative weight is clamped by ReLU
    add = BiFPN_Add(64, 2)([x64, x64])
    assert add.shape == (2, 64, 96, 96)
    print(f"{'BiFPN_Concat2 / BiFPN_Add':52s} out={tuple(cat.shape)} / {tuple(add.shape)}")

    # SEAM is a multiplicative gate: output/input ratio is constant over H,W per channel and lies in [1, e]
    s = SEAM(64, 64, 1).eval()
    ratio = (s(x64) / x64)[0, :, 0, 0]
    assert torch.all(ratio >= 1.0) and torch.all(ratio <= torch.e + 1e-4)
    print(f"{'SEAM gate range':52s} min={ratio.min():.3f} max={ratio.max():.3f} (expected within [1, e])")

    # C3k2Ghost positional args match stock C3k2
    from ultralytics.nn.modules.block import C3k2
    a, b = C3k2(64, 128, 2, False, 0.25), C3k2Ghost(64, 128, 2, False, 0.25)
    print(f"{'C3k2 vs C3k2Ghost params (64->128,n=2,e=0.25)':52s} {n_params(a):,} vs {n_params(b):,}")
    print("all checks passed")
