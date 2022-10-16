import timm
from torch import nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = timm.create_model(
            "edgenext_xx_small", pretrained=True, num_classes=0, global_pool=""
        )

        blocks = [m.stem] + [m.stages[i] for i in range(4)]
        self.blocks = nn.ModuleList(blocks)

    def extract_features(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features[1:]

    def forward(self, video):
        bsz, seq_len, c, h, w = video.shape
        imgs = video.view(bsz * seq_len, c, h, w)
        feats = self.extract_features(imgs)  # list of (bsz * seq_len, dim, h, w)
        feats = [
            f.view(bsz, seq_len, *f.shape[1:]) for f in feats
        ]  # list of (bsz, seq_len, dim, h, w). h and w are not necessarily the same.
        return feats
