import torch
import torch.nn as nn


class BiFPN_Concat2(nn.Module):
    """Weighted Bidirectional Feature Pyramid Network fusion for 2 inputs.
    Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020.

    Replaces standard Concat in the YOLO neck with fast normalized weighted fusion.
    Each input feature gets a learnable weight so the network decides how much
    each scale contributes to the fused output — unlike plain Concat which treats
    all inputs equally.

    Fast normalized fusion (paper eq.):
        O = sum(w_i / (sum(w) + eps) * Input_i)
        then concatenated along channel dim

    Args:
        dimension : concat dimension (default 1 = channel axis)
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # Learnable weights — one per input, initialised to 1 (equal importance)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001  # prevents division by zero

    def forward(self, x):
        # x is a list of 2 feature maps from the yaml multi-input layer
        w = self.w
        # Fast normalized fusion — paper section 3.3
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class BiFPN_Concat3(nn.Module):
    """Weighted BiFPN fusion for 3 inputs.
    Used when a node receives 3 feature maps (e.g. full EfficientDet BiFPN
    intermediate nodes, or extended YOLO neck variants).

    Args:
        dimension : concat dimension (default 1 = channel axis)
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)