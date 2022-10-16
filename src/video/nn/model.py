import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .layer import FFN, Layer2D, Layer3D, Squeeze, Upsample, ckpt_forward


class Decoder(nn.Module):
    def __init__(self, n_layers=1, output_dim=3, last_dim=32, n_steps=1):
        super().__init__()
        self.avg = nn.AvgPool3d(
            kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)
        )
        self.backbone = Backbone()
        self.feat_dims = [24, 48, 88, 168]
        self.num_heads = [1, 2, 4, 6]

        self.up = Upsample(2)
        layers = []
        convs = []
        for i in reversed(range(4)):
            in_dim = self.feat_dims[i]
            heads = self.num_heads[i]
            g = self.num_heads[i - 1] if i > 0 else 1
            layers.append(
                nn.Sequential(*[Layer3D(in_dim, heads) for _ in range(n_layers)])
            )

            if i == 0:
                in_dim += 3
                out_dim = last_dim
            else:
                out_dim = self.feat_dims[i - 1]
                in_dim += out_dim
            convs.append(Layer2D(in_dim=in_dim, out_dim=out_dim, groups=g))

        self.layers = nn.ModuleList(layers)
        self.convs = nn.ModuleList(convs)
        self.last_layers = nn.ModuleList(
            [
                Layer3D(last_dim, 1, without_channel_attention=True)
                for _ in range(n_layers)
            ]
        )
        self.fc = nn.Sequential(
            Layer2D(last_dim, last_dim, groups=1),
            FFN(last_dim, 1),
            Squeeze(dim=1),
            nn.Conv2d(last_dim, output_dim * n_steps, kernel_size=1),
        )
        self.n_steps = n_steps

    @ckpt_forward
    def backbone_forward(self, x):
        return self.backbone(x)

    def forward(self, video):
        size = video.shape[-2:]
        feats = [self.avg(video)]
        feats.extend(self.backbone_forward(video))
        x = feats[-1]
        for layer, conv, feat in zip(self.layers, self.convs, reversed(feats[:-1])):
            x = layer(x)
            x = self.up(x, feat)
            x = conv(x)
        for layer in self.last_layers:
            x = layer(x)
        x = self.up(x, size=size)
        x = self.fc(x[:, [-1]])
        x = x.view(x.shape[0], self.n_steps, -1, x.shape[-2], x.shape[-1])
        return x

    def loss(self, pred, gt):
        return F.binary_cross_entropy_with_logits(pred, gt)


if __name__ == "__main__":
    m = Decoder(n_layers=3, output_dim=3, last_dim=32, n_steps=5)
    h, w = (480, 720)
    x = torch.randn(1, 32, 3, h, w)
    x = x.to("cuda")
    m.to("cuda")
    y = m(x)
    print(y.shape)
