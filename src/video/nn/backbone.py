import timm
import torch
from torch import nn

from .layer import DualScaleDownsample


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = timm.create_model(
            "convnext_nano", pretrained=False, num_classes=0, global_pool=""
        )

        blocks = nn.ModuleList([m.stages[i] for i in range(4)])
        self.conv_hr = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
        self.stem = m.stem
        self.blocks = nn.ModuleList(blocks)
        self.dual_downsample = DualScaleDownsample()

    def forward(self, x):
        hr_x, lr_x = self.dual_downsample(x)
        x = torch.cat([self.conv_hr(hr_x), lr_x], dim=1)
        x = self.stem(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return hr_x, lr_x, features
