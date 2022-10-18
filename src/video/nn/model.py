import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .ckpt import ckpt_forward
from .layer import (
    ImageBlock,
    Layer2D,
    UpsampleWithRefrence,
    VideoBlock,
    ImageToVideo,
    VideoToImage,
)
from .utils import to_YCbCr420, from_YCbCr420


class Decoder(nn.Module):
    def __init__(self, last_dim=64, n_steps=1):
        super().__init__()
        self.backbone = Backbone()
        self.backbone_feat_dims = [80, 160, 320, 640]
        self.front_feat_dims = [32, 32, 64, 96]
        self.num_heads = [1, 1, 2, 3]
        self.num_layers = [1, 1, 2, 4]

        self.n_steps = n_steps

        self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        feat_blocks = []
        for i in range(4):
            in_dim = self.backbone_feat_dims[i]
            inner_dim = self.front_feat_dims[i]
            feat_blocks.append(ImageBlock(in_dim, inner_dim))
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

            for _ in range(self.num_layers[i]):
                pre_layers.append(VideoBlock(in_dim, pre_heads))

            ups.append(Layer2D(UpsampleWithRefrence(in_dim, additional_dim)))

            post_layers.append(
                Layer2D(ImageBlock(in_dim + 2 * additional_dim, out_dim))
            )
            for _ in range(self.num_layers[i]):
                post_layers.append(VideoBlock(out_dim, post_heads))

            pre_blocks.append(nn.Sequential(*pre_layers))
            post_blocks.append(nn.Sequential(*post_layers))

        self.pre_blocks = nn.ModuleList(pre_blocks)
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

    def to_YCbCr420(self, video):
        bsz = video.shape[0]
        imgs = self.to_image(video)
        l, cbcr = to_YCbCr420(imgs)
        l = self.to_video(l, bsz)
        cbcr = self.to_video(cbcr, bsz)
        return l, cbcr

    def from_YCbCr420(self, l, cbcr):
        bsz = l.shape[0]
        l = self.to_image(l)
        cbcr = self.to_image(cbcr)
        rgb = from_YCbCr420(l, cbcr)
        rgb = self.to_video(rgb, bsz)
        return rgb

    def avg(self, video):
        hr_l, cbcr = self.to_YCbCr420(video)
        l = F.avg_pool3d(hr_l, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        hr = torch.cat([l, cbcr], dim=2)
        lr = F.avg_pool3d(video, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return hr_l, hr, lr

    @ckpt_forward
    def backbone_forward(self, x):
        bsz = x.shape[0]
        x = self.to_image(x)
        l, cbcr = to_YCbCr420(x)
        feats = self.backbone(l, cbcr)
        feats = [self.feat_blocks[i](feat) for i, feat in enumerate(feats)]
        feats = [self.to_video(feat, bsz) for feat in feats]
        return feats

    def forward(self, video):
        l, hr, lr = self.avg(video)
        feats = [lr]
        feats.extend(self.backbone_forward(video))
        x = feats[-1]
        for pre, up, post, feat in zip(
            self.pre_blocks, self.ups, self.post_blocks, reversed(feats[:-1])
        ):
            x = pre(x)
            x = up(x, feat)
            x = post(x)

        x = self.last_up1(x[:, -1], hr[:, -1])
        x = self.refine1(x)
        cbcr = self.fc_cbcr(x).sigmoid()
        x = self.last_up2(x, l[:, -1])
        x = self.refine2(x)
        l = self.fc_l(x).sigmoid()

        l = l.view(l.shape[0], self.n_steps, -1, l.shape[2], l.shape[3])
        cbcr = cbcr.view(cbcr.shape[0], self.n_steps, -1, cbcr.shape[2], cbcr.shape[3])
        return l, cbcr

    def loss(self, pred_l, pred_cbcr, gold):
        gt_l, gt_cbcr = self.to_YCbCr420(gold)
        loss_l = F.l1_loss(pred_l, gt_l)
        loss_cbcr = F.l1_loss(pred_cbcr, gt_cbcr) * 2
        return (loss_l + loss_cbcr) / 3


if __name__ == "__main__":
    m = Decoder(n_layers=3, output_dim=3, last_dim=32, n_steps=5)
    h, w = (480, 720)
    x = torch.randn(1, 32, 3, h, w)
    x = x.to("cuda")
    m.to("cuda")
    y = m(x)
    print(y.shape)
