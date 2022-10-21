import torch
from torch import nn

from .layer import (
    DualScaleUpsample,
    ImageToVideo,
    Layer2D,
    Sigmoid,
    Stage,
    UpsampleWithRefrence,
    VideoBlock,
    VideoToImage,
)


class Encoder(nn.Module):
    def __init__(self, backbone_feat_dims, front_feat_dims, num_heads, num_layers):
        super().__init__()

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

        feat_time_blocks = []
        for i in range(4):
            backbone_feat_dims[i]
            inner_dim = front_feat_dims[i]
            heads = num_heads[i]
            feat_time_blocks.append(VideoBlock(inner_dim, heads, num_layers[i]))
        self.feat_time_blocks = nn.ModuleList(feat_time_blocks)

    def forward(self, feats, bsz):
        feats = [self.to_video(feat, bsz) for feat in feats]
        feats = [self.feat_time_blocks[i](feat) for i, feat in enumerate(feats)]
        return feats


class Decoder(nn.Module):
    def __init__(self, feat_dims, num_heads, num_layers, last_dim, n_steps):
        super().__init__()
        self.n_steps = n_steps

        merges = []
        blocks = []
        prev_dim = feat_dims[-1]
        for i in reversed(range(4)):
            additional_dim = feat_dims[i - 1] if i > 0 else 3
            out_dim = last_dim if i == 0 else feat_dims[i - 1]
            heads = num_heads[i]

            blocks.append(
                nn.Sequential(
                    Layer2D(Stage(prev_dim, out_dim, num_layers[i], mode="up")),
                    VideoBlock(out_dim, heads, num_layers[i]),
                )
            )
            merges.append(Layer2D(UpsampleWithRefrence(out_dim, additional_dim)))
            prev_dim = out_dim + 2 * additional_dim

        self.merges = nn.ModuleList(merges)
        self.blocks = nn.ModuleList(blocks)

        self.refine_lr = Stage(prev_dim, last_dim, 3, mode="up")
        self.last_up_lr = UpsampleWithRefrence(last_dim, 3)
        self.fc_lr = nn.Conv2d(last_dim + 6, 3 * self.n_steps, 1)

        self.refine_hr = Stage(last_dim + 6, last_dim, 3, mode="up")
        self.last_up_hr = UpsampleWithRefrence(last_dim, 1)
        self.fc_hr = nn.Conv2d(last_dim + 1, 1 * self.n_steps, 1)

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

        self.avg_pool = Layer2D(nn.AvgPool2d(2, 2))
        self.sigmoid = Sigmoid()

        self.dual_upsample = DualScaleUpsample()

    def duplicate_last(self, x):
        return torch.cat([x, x[:, [-1]]], dim=1)

    def forward(self, video, hr_video, lr_video, feats):
        b = video.shape[0]
        h, w = video.shape[-2:]
        video = self.duplicate_last(video)
        hr_video = self.duplicate_last(hr_video)
        lr_video = self.duplicate_last(lr_video)
        feats = [self.duplicate_last(feat) for feat in feats]

        avg_video = self.avg_pool(lr_video)
        feats = [avg_video] + feats
        x = feats[-1]
        for merge, block, feat in zip(self.merges, self.blocks, reversed(feats[:-1])):
            x = block(x)
            x = merge(x, feat)

        x = self.refine_lr(x[:, -1])
        x = self.last_up_lr(x, lr_video[:, -1].contiguous())
        lr_x = self.fc_lr(x)

        x = self.refine_hr(x)
        x = self.last_up_hr(x, hr_video[:, -1].contiguous())
        hr_x = self.fc_hr(x)

        lr_x = lr_x.reshape(b * self.n_steps, 3, h // 2, w // 2)
        hr_x = hr_x.reshape(b * self.n_steps, 1, h, w)
        x = self.dual_upsample(hr_x, lr_x)
        x = x.view(b, self.n_steps, 3, h, w)
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

    def backbone_forward(self, x):
        hr_x, lr_x, feats = self.backbone(x)
        return hr_x, lr_x, feats

    def encode(self, video):
        bsz = video.shape[0]
        x = self.to_image(video)
        hr_x, lr_x, feats = self.backbone_forward(x)
        hr_x = self.to_video(hr_x, bsz)
        lr_x = self.to_video(lr_x, bsz)
        feats = self.encoder(feats, bsz)
        return hr_x, lr_x, feats

    def __call__(self, video):
        hr_video, lr_video, feats = self.encode(video)
        rgb = self.decoder(video, hr_video, lr_video, feats)
        return rgb
