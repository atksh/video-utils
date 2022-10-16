import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .layer import Layer2D, Layer3D, Upsample, ckpt_forward


class Decoder(nn.Module):
    def __init__(self, n_layers=1, output_dim=3, last_dim=32, n_steps=1):
        super().__init__()
        self.avg = nn.AvgPool3d(kernel_size=(1, 2, 2), padding=0, stride=(1, 2, 2))
        self.backbone = Backbone()
        self.feat_dims = [24, 48, 88, 168]
        self.num_heads = [1, 2, 4, 6]

        self.up = Upsample(2)
        layers = []
        convs = []
        for i in reversed(range(4)):
            in_dim = self.feat_dims[i]
            heads = self.num_heads[i]
            layers.append(
                nn.Sequential(*[Layer3D(in_dim, heads) for _ in range(n_layers)])
            )
            if i == 0:
                in_dim += 3
                out_dim = last_dim
            else:
                out_dim = self.feat_dims[i - 1]
                in_dim += out_dim
            convs.append(
                nn.Sequential * [Layer2D(in_dim, out_dim) for _ in range(n_layers)]
            )

        self.layers = nn.ModuleList(layers)
        self.convs = nn.ModuleList(convs)

        self.post = nn.Sequential(
            Layer2D(last_dim + 3, last_dim),
            Layer3D(last_dim, 1),
        )
        self.fc = nn.Conv2d(last_dim, output_dim * n_steps, kernel_size=1)
        self.n_steps = n_steps

    @ckpt_forward
    def backbone_forward(self, x):
        return self.backbone(x)

    def forward(self, video):
        feats = [self.avg(video)]
        feats.extend(self.backbone_forward(video))
        x = feats[-1]
        for layer, conv, feat in zip(self.layers, self.convs, reversed(feats[:-1])):
            x = layer(x)
            x = self.up(x, feat)
            x = conv(x)
        x = self.up(x, video)
        x = self.post(x)
        x = self.fc(x[:, -1])
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
