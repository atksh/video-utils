import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .ckpt import ckpt_forward
from .layer import (
    ImageBlock,
    ImageToVideo,
    Layer2D,
    UpsampleWithRefrence,
    VideoBlock,
    VideoToImage,
)
from .utils import RGB2YCbCr420, YCbCr420ToRGB, from_YCbCr420, soft_clip, to_YCbCr420


class Encoder(nn.Module):
    def __init__(self, backbone_feat_dims, front_feat_dims, num_heads, num_layers):
        super().__init__()

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

        self.backbone = Backbone()
        feat_blocks = []
        feat_time_blocks = []
        for i in range(4):
            in_dim = backbone_feat_dims[i]
            inner_dim = front_feat_dims[i]
            heads = num_heads[i]
            feat_blocks.append(ImageBlock(in_dim, inner_dim))
            feat_time_blocks.append(
                nn.Sequential(
                    *[VideoBlock(inner_dim, heads) for _ in range(num_layers[i])]
                )
            )
        self.feat_blocks = nn.ModuleList(feat_blocks)
        self.feat_time_blocks = nn.ModuleList(feat_time_blocks)

    def forward(self, x):
        bsz = x.shape[0]
        x = self.to_image(x)
        l, cbcr = to_YCbCr420(x)
        feats = self.backbone(l, cbcr)
        feats = [self.feat_blocks[i](feat) for i, feat in enumerate(feats)]
        feats = [self.to_video(feat, bsz) for feat in feats]
        feats = [self.feat_time_blocks[i](feat) for i, feat in enumerate(feats)]
        return feats


class Decoder(nn.Module):
    def __init__(self, feat_dims, last_dim, n_steps):
        super().__init__()
        self.n_steps = n_steps

        ups = []
        post_blocks = []
        for i in reversed(range(4)):
            in_dim = feat_dims[i]
            additional_dim = feat_dims[i - 1] if i > 0 else 3
            out_dim = last_dim if i == 0 else feat_dims[i - 1]

            ups.append(Layer2D(UpsampleWithRefrence(in_dim, additional_dim)))

            post_blocks.append(
                Layer2D(
                    nn.Sequential(
                        ImageBlock(in_dim + 2 * additional_dim, out_dim),
                        ImageBlock(out_dim, out_dim),
                    )
                )
            )

        self.ups = nn.ModuleList(ups)
        self.post_blocks = nn.ModuleList(post_blocks)

        self.last_up1 = UpsampleWithRefrence(last_dim, 3)
        self.refine1 = ImageBlock(last_dim + 6, last_dim)
        self.fc_cbcr = nn.Conv2d(last_dim, 2 * self.n_steps, 1)

        self.last_up2 = UpsampleWithRefrence(last_dim, 1)
        self.refine2 = ImageBlock(last_dim + 2, last_dim)
        self.fc_l = nn.Conv2d(last_dim, 1 * self.n_steps, 1)

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()
        self.to_YCbCr420 = RGB2YCbCr420()

        self.avg_pool = Layer2D(nn.AvgPool2d(2, 2))

    def avg(self, video):
        hr_l, cbcr = self.to_YCbCr420(video)
        l = self.avg_pool(hr_l)
        hr = torch.cat([l, cbcr], dim=2)
        lr = self.avg_pool(hr)
        return hr_l, hr, lr

    def post_process(self, l, cbcr):
        l = torch.sign(l) * torch.log1p(torch.abs(l))
        cbcr = torch.sign(cbcr) * torch.log1p(torch.abs(cbcr))
        l = soft_clip(l, -1, 1) * 0.5 + 0.5
        cbcr = soft_clip(cbcr, -1, 1) * 0.5 + 0.5
        return l, cbcr

    def forward(self, video, feats):
        l, hr, lr = self.avg(video)
        feats = [lr] + feats
        x = feats[-1]
        for up, post, feat in zip(self.ups, self.post_blocks, reversed(feats[:-1])):
            x = up(x, feat)
            x = post(x)

        x = self.last_up1(x[:, -1], hr[:, -1])
        x = self.refine1(x)
        cbcr = self.fc_cbcr(x)
        x = self.last_up2(x, l[:, -1])
        x = self.refine2(x)
        l = self.fc_l(x)

        l = l.view(l.shape[0], self.n_steps, -1, l.shape[2], l.shape[3])
        cbcr = cbcr.view(cbcr.shape[0], self.n_steps, -1, cbcr.shape[2], cbcr.shape[3])
        l, cbcr = self.post_process(l, cbcr)
        return l, cbcr


class EncDecModel(nn.Module):
    def __init__(
        self,
        backbone_feat_dims,
        front_feat_dims,
        num_heads,
        num_layers,
        last_dim,
        n_steps,
    ):
        super().__init__()
        self.encoder = Encoder(
            backbone_feat_dims, front_feat_dims, num_heads, num_layers
        )
        self.decoder = Decoder(front_feat_dims, last_dim, n_steps)
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()
        self.to_YCbCr420 = RGB2YCbCr420()
        self.from_YCbCr420 = YCbCr420ToRGB()

    def encode(self, video):
        return self.encoder(video)

    def decode(self, video, feats):
        return self.decoder(video, feats)

    def forward(self, video):
        feats = self.encode(video)
        l, cbcr = self.decode(video, feats)
        return l, cbcr

    def inference(self, video):
        l, cbcr = self.forward(video)
        rgb = self.from_YCbCr420(l, cbcr)
        return rgb

    def loss(self, pred_l, pred_cbcr, gold):
        gt_l, gt_cbcr = self.to_YCbCr420(gold)
        loss_l = F.l1_loss(pred_l, gt_l)
        loss_cbcr = F.l1_loss(pred_cbcr, gt_cbcr) * 2
        return (loss_l + loss_cbcr) / 3
