from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from .layer import DualScaleDownsample


class LayerScaler(nn.Module):
    def __init__(self, s: float, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(s * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        return self.gamma[None, ..., None, None] * x


class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class SELayer(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        intermed_dim = max(dim // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(dim, intermed_dim),
            nn.SiLU(),
            nn.Linear(intermed_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        block_size: int,
        layer_scaler_init_value=1e-6,
        drop_p=0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            dim,
            dim,
            groups=dim,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.ln = LayerNorm2D(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.SiLU(),
            SELayer(dim * 4),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, dim)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

    def forward(self, x):
        resid = x
        x = self.conv(x)
        x = self.ln(x)
        x = self.mlp(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += resid
        return x


class Stage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        window_size: int,
        block_size: int,
        depth: int,
        drop_p: float,
    ):
        super().__init__()
        self.down = nn.Sequential(
            LayerNorm2D(in_dim), nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        )
        self.blocks = nn.Sequential(
            *[
                Block(out_dim, window_size, block_size, drop_p=drop_p)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        return x


class Backbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        stem_dim: int,
        widths: List[int],
        depths: List[int],
        window_sizes: List[int],
        block_sizes: List[int],
        drop_p: float = 0.25,
    ):
        super().__init__()
        self.dual_downsample = DualScaleDownsample()
        self.conv_hr = nn.Conv2d(1, 3, kernel_size=2, stride=2, bias=False)
        self.stem = nn.Conv2d(in_dim, stem_dim, kernel_size=4, stride=4, bias=False)

        in_out_widths = [(stem_dim, widths[0])] + list(zip(widths[:-1], widths[1:]))
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        stages = []
        for (in_dim, out_dim), depth, drop_p, window_size, block_size in zip(
            in_out_widths, depths, drop_probs, window_sizes, block_sizes
        ):
            stages.append(
                Stage(
                    in_dim,
                    out_dim,
                    window_size=window_size,
                    block_size=block_size,
                    depth=depth,
                    drop_p=drop_p,
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        hr_x, lr_x = self.dual_downsample(x)
        x = self.conv_hr(hr_x) + lr_x
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return hr_x, lr_x, feats
