import timm
import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = timm.create_model(
            "convnext_nano", pretrained=True, num_classes=0, global_pool=""
        )

        blocks = nn.ModuleList([m.stages[i] for i in range(4)])
        self.conv_l = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
        self.stem = m.stem
        self.blocks = nn.ModuleList(blocks)

    def extract_features(self, l, cbcr):
        l = self.conv_l(l)
        x = self.stem(torch.cat([l, cbcr], dim=1))
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def forward(self, l, cbcr):
        feats = self.extract_features(l, cbcr)
        return feats
