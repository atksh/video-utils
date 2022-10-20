import torch
from torch import nn
from torch.nn import functional as F

from .ckpt import ckpt_forward
from .layer import (
    DualScaleDownsample,
    DualScaleUpsample,
    ImageReduction,
    ImageReduction3D,
    ImageToVideo,
    Layer2D,
    Sigmoid,
    UpsampleWithRefrence,
    VideoBlock,
    VideoToImage,
)


class Encoder(nn.Module):
    def __init__(self, backbone_feat_dims, front_feat_dims, num_heads, num_layers):
        super().__init__()

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

        feat_blocks = []
        feat_time_blocks = []
        for i in range(4):
            in_dim = backbone_feat_dims[i]
            inner_dim = front_feat_dims[i]
            heads = num_heads[i]
            feat_blocks.append(ImageReduction(in_dim, inner_dim, num_layers[1]))
            feat_time_blocks.append(VideoBlock(inner_dim, heads, num_layers[i]))
        self.feat_blocks = nn.ModuleList(feat_blocks)
        self.feat_time_blocks = nn.ModuleList(feat_time_blocks)

    def forward(self, feats, bsz):
        feats = [self.feat_blocks[i](feat) for i, feat in enumerate(feats)]
        feats = [self.to_video(feat, bsz) for feat in feats]
        feats = [self.feat_time_blocks[i](feat) for i, feat in enumerate(feats)]
        return feats


class Decoder(nn.Module):
    def __init__(self, feat_dims, num_heads, num_layers, last_dim, n_steps):
        super().__init__()
        self.n_steps = n_steps

        ups = []
        post_blocks = []
        for i in reversed(range(4)):
            in_dim = feat_dims[i]
            additional_dim = feat_dims[i - 1] if i > 0 else 3
            out_dim = last_dim if i == 0 else feat_dims[i - 1]
            heads = num_heads[i]

            ups.append(Layer2D(UpsampleWithRefrence(in_dim, additional_dim)))

            cat_dim = in_dim + 2 * additional_dim
            post_blocks.append(
                nn.Sequential(
                    ImageReduction3D(cat_dim, out_dim, num_layers[i]),
                    VideoBlock(out_dim, heads, num_layers[i]),
                )
            )

        self.ups = nn.ModuleList(ups)
        self.post_blocks = nn.ModuleList(post_blocks)

        self.last_up_lr = UpsampleWithRefrence(last_dim, 3)
        self.refine_lr = ImageReduction(last_dim + 6, last_dim, 3)
        self.fc_lr = nn.Conv2d(last_dim, 2 * self.n_steps, 1)

        self.last_up_hr = UpsampleWithRefrence(last_dim, 1)
        self.refine_hr = ImageReduction(last_dim + 2, last_dim, 3)
        self.fc_hr = nn.Conv2d(last_dim, 1 * self.n_steps, 1)

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

        self.avg_pool = Layer2D(nn.AvgPool2d(2, 2))
        self.sigmoid = Sigmoid()

        self.dual_downsample = Layer2D(DualScaleDownsample())
        self.dual_upsample = Layer2D(DualScaleUpsample())

    def duplicate_last(self, x):
        return torch.cat([x, x[:, [-1]]], dim=1)

    def forward(self, video, feats):
        video = self.duplicate_last(video)
        feats = [self.duplicate_last(feat) for feat in feats]

        hr_video, lr_video = self.dual_downsample(video)
        avg_video = self.avg_pool(lr_video)
        feats = [avg_video] + feats
        x = feats[-1]
        for up, post, feat in zip(self.ups, self.post_blocks, reversed(feats[:-1])):
            x = up(x, feat)
            x = post(x)

        x = self.last_up_lr(x[:, -1], lr_video[:, -1])
        x = self.refine_lr(x)
        lr_x = self.fc_lr(x)
        x = self.last_up_hr(x, hr_video[:, -1])
        x = self.refine_hr(x)
        hr_x = self.fc_hr(x)

        lr_x = lr_x.view(lr_x.shape[0], self.n_steps, 2, lr_x.shape[2], lr_x.shape[3])
        hr_x = hr_x.view(hr_x.shape[0], self.n_steps, 1, hr_x.shape[2], hr_x.shape[3])
        x = self.dual_upsample(hr_x, lr_x)
        x = self.sigmoid(x)
        return x


class MergedModel:
    def __init__(
        self,
        backbone: nn.Module,
        encoder: Encoder,
        decoder: Decoder,
    ):
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    @ckpt_forward
    def backbone_forward(self, x):
        bsz = x.shape[0]
        hr_x, lr_x = self.decoder.dual_downsample(x)
        hr_x = self.to_image(hr_x)
        lr_x = self.to_image(lr_x)
        feats = self.backbone(hr_x, lr_x)
        return feats, bsz

    def encode(self, video):
        feats, bsz = self.backbone_forward(video)
        return self.encoder(feats, bsz)

    def decode(self, video, feats):
        return self.decoder(video, feats)

    def __call__(self, video):
        feats = self.encode(video)
        rgb = self.decode(video, feats)
        return rgb
