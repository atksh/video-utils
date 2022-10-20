import timm
import torch
from torch import nn


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

    def forward(self, hr_x, lr_x):
        x = torch.cat([self.conv_hr(hr_x), lr_x], dim=1)
        x = self.stem(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
