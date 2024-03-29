import torch
from torch import nn
from torch.nn import functional as F

from .freq_layer import FreqBackbone
from .layer import ImageToVideo, Layer2D, Stage, VideoBlock, VideoToImage


class Encoder(nn.Module):
    def __init__(self, in_dim, widths, depths, heads):
        super().__init__()
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()
        self.backbone = FreqBackbone(
            in_ch=in_dim,
            depths=depths,
            widths=widths,
            block_size=8,
            n=8,
            return_freq=False,
        )

        feat_time_blocks = []
        for i in range(len(widths)):
            inner_dim = widths[i]
            feat_time_blocks.append(VideoBlock(inner_dim, heads[i], depths[i]))
        self.feat_time_blocks = nn.ModuleList(feat_time_blocks)

    def forward(self, video):
        # (B, T, C, H, W) -> list of (B, T, C, H // k, W // k) k = 4, 8, ...
        bsz = video.shape[0]
        x = self.to_image(video)
        feats = self.backbone(x)
        feats = [self.to_video(feat, bsz) for feat in feats]
        feats = [self.feat_time_blocks[i](feat) for i, feat in enumerate(feats)]
        return feats


class Decoder(nn.Module):
    def __init__(self, out_dim, last_dim, widths, depths, heads):
        super().__init__()
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

        up_blocks = []
        refine_blocks = []
        for i in reversed(range(len(widths) - 1)):
            lr_dim = widths[i + 1]
            hr_dim = widths[i]
            cat_dim = lr_dim + hr_dim
            up_blocks.append(
                Layer2D(nn.ConvTranspose2d(lr_dim, lr_dim, kernel_size=2, stride=2))
            )
            refine_blocks.append(
                nn.Sequential(
                    Layer2D(Stage(cat_dim, hr_dim, 1, mode="same")),
                    VideoBlock(hr_dim, heads[i], depths[i]),
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.refine_blocks = nn.ModuleList(refine_blocks)

        self.stem = nn.Conv2d(3, last_dim, 2, stride=2, bias=False)
        self.last_up = Stage(last_dim, last_dim, 1, mode="up")
        self.fc = nn.Conv2d(last_dim * 2, out_dim, 1)

    def resize_like(self, x, ref):
        bsz = None
        size = ref.shape[-2:]
        if x.ndim == 5:
            bsz = x.shape[0]
            x = self.to_image(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        if bsz is not None:
            x = self.to_video(x, bsz)
        return x

    def duplicate_last(self, x):
        return torch.cat([x, x[:, [-1]]], dim=1)

    def soft_clip(self, x, lb, ub):
        _x = torch.clamp(x, lb, ub)
        return _x.detach() + x - x.detach()

    def get_ref(self, last_video, cord):
        cord_x = cord[:, 0].softmax(dim=-1).cumsum(dim=-1)
        cord_y = cord[:, 1].softmax(dim=-2).cumsum(dim=-2)
        cord = torch.stack([cord_x, cord_y], dim=-1) * 2 - 1
        ref = F.grid_sample(last_video, cord, mode="bilinear", align_corners=True)
        return ref

    def forward(self, video, feats):
        feats = [self.duplicate_last(feat) for feat in feats]
        last_video = video[:, -1].contiguous()

        x = feats[-1]
        for up_block, refine_block, feat in zip(
            self.up_blocks, self.refine_blocks, reversed(feats[:-1])
        ):
            x = up_block(x)
            x = self.resize_like(x, feat)
            x = torch.cat([x, feat], dim=2)
            x = refine_block(x)
        x = x[:, -1].contiguous()
        # now x is (B, C, H // 4, W // 4)
        z = self.stem(last_video)
        x = self.last_up(x)
        x = self.resize_like(x, z)
        x = torch.cat([x, z], dim=1)
        # now x is (B, C, H // 2, W // 2)
        x = self.fc(x)
        x = self.resize_like(x, last_video)
        # now x is (B, C, H, W)
        x = self.soft_clip(x, 0, 1).unsqueeze(1)
        return x
