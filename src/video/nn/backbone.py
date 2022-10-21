from typing import List

import torch
from torch import nn

from .layer import DualScaleDownsample, Stage


class Backbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        stem_dim: int,
        widths: List[int],
        depths: List[int],
        drop_p: float = 0.25,
    ):
        super().__init__()
        self.dual_downsample = DualScaleDownsample()
        self.conv_hr = nn.Conv2d(1, 3, kernel_size=2, stride=2, bias=False)
        self.stem = nn.Conv2d(in_dim, stem_dim, kernel_size=4, stride=4, bias=False)

        in_out_widths = [(stem_dim, widths[0])] + list(zip(widths[:-1], widths[1:]))
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        stages = []
        for (in_dim, out_dim), depth, drop_p in zip(in_out_widths, depths, drop_probs):
            stages.append(
                Stage(
                    in_dim,
                    out_dim,
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
