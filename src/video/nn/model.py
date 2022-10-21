import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .layer import ImageToVideo, Layer2D, Stage, VideoBlock, VideoToImage


class Encoder(nn.Module):
    def __init__(self, in_dim, stem_dim, widths, depths, heads, drop_p=0.0):
        super().__init__()
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()
        self.backbone = Backbone(in_dim, stem_dim, widths, depths, drop_p)

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
            up_blocks.append(Layer2D(Stage(lr_dim, hr_dim, depths[i], mode="up")))
            refine_blocks.append(
                VideoBlock(hr_dim, heads[i], depths[i]),
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.refine_blocks = nn.ModuleList(refine_blocks)

        self.stem = nn.Conv2d(3, last_dim, 2, stride=2, bias=False)
        self.last_up = Stage(last_dim, last_dim, 1, mode="up")
        self.mlp = nn.Sequential(
            nn.Conv2d(last_dim, last_dim, 3, padding=1),
            nn.GroupNorm(1, last_dim),
            nn.SiLU(),
            nn.Conv2d(last_dim, out_dim + 3 + 2, 3, padding=1),
        )

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

    def forward(self, video, feats):
        feats = [self.duplicate_last(feat) for feat in feats]
        last_video = video[:, -1].contiguous()

        x = feats[-1]
        for up_block, refine_block, feat in zip(
            self.up_blocks, self.refine_blocks, reversed(feats[:-1])
        ):
            x = up_block(x)
            x = self.resize_like(x, feat) + feat
            x = refine_block(x)
        x = x[:, -1].contiguous()
        # now x is (B, C, H // 4, W // 4)
        z = self.stem(last_video)
        x = self.last_up(x)
        x = self.resize_like(x, z) + z
        # now x is (B, C, H // 2, W // 2)
        x = self.mlp(x)
        x = self.resize_like(x, last_video)
        # now x is (B, C, H, W)
        x, s, cord = x.chunk(3, dim=1)
        cord_x = cord[:, 0].softmax(dim=-1).cumsum(dim=-1)
        cord_y = cord[:, 1].softmax(dim=-2).cumsum(dim=-2)
        cord = torch.stack([cord_x, cord_y], dim=-1) * 2 - 1
        ref = F.grid_sample(last_video, cord, mode="bilinear", align_corners=True)

        s = torch.cat(s.softmax(dim=1).chunk(3, dim=1), dim=0)
        x = torch.stack([x, ref, last_video], dim=0)
        x = (x * s).sum(dim=0)
        x = self.soft_clip(x, 0, 1)
        return x.unsqueeze(1)
