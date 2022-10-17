from tkinter import Image

import torch
from torch import nn

from .backbone import Backbone
from .ckpt import ckpt_forward
from .layer import ImageBlock, Layer2D, UpsampleWithRefrence, VideoBlock
from .utils import DiscMixLogistic


class Decoder(nn.Module):
    def __init__(self, n_layers=1, last_dim=64, n_steps=1, num_mix=4, num_bits=8):
        super().__init__()
        self.avg = nn.AvgPool3d(kernel_size=(1, 2, 2), padding=0, stride=(1, 2, 2))
        self.backbone = Backbone()
        self.backbone_feat_dims = [80, 160, 320, 640]
        self.front_feat_dims = [128, 192, 288, 384]
        self.num_heads = [4, 6, 9, 12]

        self.n_steps = n_steps
        self.num_mix = num_mix
        self.num_bits = num_bits
        self.output_dim = 10 * num_mix

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        feat_blocks = []
        for i in range(4):
            in_dim = self.backbone_feat_dims[i]
            inner_dim = self.front_feat_dims[i]
            feat_blocks.append(Layer2D(ImageBlock(in_dim, inner_dim)))
        self.feat_blocks = nn.ModuleList(feat_blocks)

        pre_blocks = []
        ups = []
        post_blocks = []
        for i in reversed(range(4)):
            pre_layers = []
            post_layers = []

            in_dim = self.front_feat_dims[i]
            additional_dim = self.front_feat_dims[i - 1] if i > 0 else 3
            out_dim = last_dim if i == 0 else self.front_feat_dims[i - 1]
            pre_heads = self.num_heads[i]
            post_heads = self.num_heads[i - 1] if i > 0 else 1

            for _ in range(n_layers):
                pre_layers.append(VideoBlock(in_dim, pre_heads))
                pre_layers.append(Layer2D(ImageBlock(in_dim, in_dim)))

            ups.append(Layer2D(UpsampleWithRefrence(in_dim, additional_dim)))

            post_layers.append(
                Layer2D(ImageBlock(in_dim + 2 * additional_dim, out_dim))
            )
            for _ in range(n_layers):
                post_layers.append(VideoBlock(out_dim, post_heads))
                post_layers.append(Layer2D(ImageBlock(out_dim, out_dim)))

            pre_blocks.append(nn.Sequential(*pre_layers))
            post_blocks.append(nn.Sequential(*post_layers))

        self.pre_blocks = nn.ModuleList(pre_blocks)
        self.ups = nn.ModuleList(ups)
        self.post_blocks = nn.ModuleList(post_blocks)

        self.last_up = UpsampleWithRefrence(last_dim, 3)
        self.fc = nn.Conv2d(last_dim + 6, self.output_dim * self.n_steps, kernel_size=1)

    @ckpt_forward
    def backbone_forward(self, x):
        feats = self.backbone(x)
        return [self.feat_blocks[i](feat) for i, feat in enumerate(feats)]

    def forward(self, video):
        feats = [self.avg(video)]
        feats.extend(self.backbone_forward(video))
        x = feats[-1]
        for pre, up, post, feat in zip(
            self.pre_blocks, self.ups, self.post_blocks, reversed(feats[:-1])
        ):
            x = pre(x)
            x = up(x, feat)
            x = post(x)

        x = self.last_up(x[:, -1], video[:, -1])
        x = self.fc(x)
        x = x.view(x.shape[0], self.n_steps, -1, x.shape[-2], x.shape[-1])
        out = []
        for i in range(self.n_steps):
            out.append(DiscMixLogistic(x[:, i], self.num_mix, self.num_bits))
        return out

    def loss(self, pred, gt):
        loss = 0.0
        for i in range(self.n_steps):
            y = gt[:, i]
            loss += pred[i].loss(y)
        return loss / self.n_steps


if __name__ == "__main__":
    m = Decoder(n_layers=3, output_dim=3, last_dim=32, n_steps=5)
    h, w = (480, 720)
    x = torch.randn(1, 32, 3, h, w)
    x = x.to("cuda")
    m.to("cuda")
    y = m(x)
    print(y.shape)
