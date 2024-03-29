import revlib
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth

from .fuse import aot_fuse

NEG_INF = -5000.0


class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


class Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


class NonLinear(nn.Module):
    def forward(self, x):
        return F.silu(x)


class _ReversibleSequential(nn.Module):
    def __init__(self, layers, split_dim):
        super().__init__()
        self.split_dim = split_dim
        self.layers = revlib.ReversibleSequential(*layers, split_dim=split_dim)
        self.s = nn.Parameter(torch.zeros(1))
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = torch.cat([x, x], dim=self.split_dim)
        x1, x2 = self.layers(x).chunk(2, dim=self.split_dim)
        s = self.sigmoid(self.s)
        x = s * x1 + (1 - s) * x2
        return x


class Sequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def ReversibleSequential(layers, split_dim):
    if len(layers) <= 4:
        return aot_fuse(Sequential(layers))
    else:
        layers = [aot_fuse(layer) for layer in layers]
        return _ReversibleSequential(layers, split_dim)


class VideoToImage(nn.Module):
    def forward(self, x):
        b, t, c, h, w = x.shape
        return x.view(b * t, c, h, w)


class ImageToVideo(nn.Module):
    def forward(self, x, batch_size: int):
        _, c, h, w = x.shape
        return x.view(batch_size, -1, c, h, w)


class LayerScaler(nn.Module):
    def __init__(self, s: float, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(s * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        return self.gamma[None, ..., None, None] * x


class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class SELayer(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        intermed_dim = max(dim // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(dim, intermed_dim),
            NonLinear(),
            nn.Linear(intermed_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FFN(nn.Module):
    def __init__(self, dim, s=2):
        super().__init__()
        self.wi = nn.Conv2d(
            dim,
            dim * s * 2,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.wo = nn.Conv2d(
            dim * s,
            dim,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.act = NonLinear()

    def forward(self, x):
        x1, x2 = self.wi(x).chunk(2, dim=1)
        x = x1 * self.act(x2)
        x = self.wo(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scaler_init_value=1.0,
        drop_p=0.0,
    ):
        super().__init__()
        conv = nn.Conv2d(
            dim,
            dim,
            groups=dim,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.layer = nn.Sequential(
            conv,
            LayerNorm2D(dim),
            FFN(dim),
            SELayer(dim),
            LayerScaler(layer_scaler_init_value, dim),
            StochasticDepth(drop_p, mode="batch"),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Stage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        drop_p: float = 0.0,
        mode="down",
    ):
        super().__init__()
        if mode == "down":
            self.ln = LayerNorm2D(in_dim)
            self.up_or_down = nn.Conv2d(in_dim, out_dim, 2, stride=2, bias=False)
        elif mode == "up":
            self.ln = LayerNorm2D(in_dim)
            self.up_or_down = nn.ConvTranspose2d(
                in_dim, out_dim, 2, stride=2, bias=False
            )
        elif mode == "same":
            self.ln = nn.Identity()
            if in_dim == out_dim:
                self.up_or_down = nn.Identity()
            else:
                self.up_or_down = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be one of 'down', 'up', 'same'."
            )

        self.blocks = ReversibleSequential(
            [Block(out_dim, drop_p=drop_p) for _ in range(depth)], split_dim=1
        )

    def forward(self, x):
        x = self.ln(x)
        x = self.up_or_down(x)
        x = self.blocks(x)
        return x


class Layer2D(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    def forward(self, *args):
        bsz = args[0].shape[0]
        new_args = [self.to_image(arg) for arg in args]
        x = self.module(*new_args)
        x = self.to_video(x, bsz)
        return x


class SimpleLayer2D(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    def forward(self, x):
        bsz = x.shape[0]
        x = self.to_image(x)
        x = self.module(x)
        x = self.to_video(x, bsz)
        return x


# 3D layers


def VideoLayerNorm(dim, eps=1e-5):
    return SimpleLayer2D(LayerNorm2D(dim, eps))


class ConvGRU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_z = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.conv_r = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.conv_h = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, video):
        # video: (batch_size, seq_len, dim, height, width)
        seq_len = video.shape[1]
        out = []
        h = torch.zeros_like(video[:, 0])
        for i in range(seq_len):
            x = video[:, i]
            xh = torch.cat([x, h], dim=1)
            z = self.sigmoid(self.conv_z(xh))
            r = self.sigmoid(self.conv_r(xh))
            xh = torch.cat([x, r * h], dim=1)
            h = (1 - z) * h + z * self.tanh(self.conv_h(xh))
            out.append(h)
        out = torch.stack(out, dim=1)
        return out


class TimeConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 2, bias=False, padding=0)

    def forward(self, video):
        b, t, c, h, w = video.shape
        # b t c h w -> (b h w) c t"
        video = video.permute(0, -1, -2, 1, 2).contiguous().view(-1, c, t)
        video = F.pad(video, (1, 0))
        video = self.conv(video)
        # (b h w) c t -> b t c h w, h=h, w=w, b=b
        video = video.view(b, h, w, c, t)
        video = video.permute(0, -1, -2, 1, 2)
        return video


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            NonLinear(),
            nn.Linear(dim, dim),
            NonLinear(),
            nn.Linear(dim, dim),
        )
        self.ln = VideoLayerNorm(dim)

    def forward(self, video):
        l = video.shape[1]
        t = torch.linspace(0, 1, l, device=video.device).view(l, 1)
        emb = self.mlp(t).view(1, l, -1, 1, 1)
        return self.ln(video + emb)


class ChannelVideoAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)

    def forward(self, q, k, v):
        bsz = q.shape[0]
        len_s = q.shape[1]
        len_t = k.shape[1]
        height, width = q.shape[-2:]
        q = self.Q(q.mean(dim=[-1, -2]))  # (batch_size, len_s, dim)
        k = self.K(k.mean(dim=[-1, -2]))  # (batch_size, len_t, dim)
        # b m c h w -> b (h w) m c
        v = v.permute(0, -1, -2, 1, 2).contiguous()
        v = v.view(bsz, height * width, len_t, -1)
        v = self.V(v)

        q = q.view(bsz, len_s, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(bsz, len_t, self.heads, -1).permute(0, 2, 3, 1)
        v = v.view(bsz, height * width, len_t, self.heads, -1)
        v = v.permute(0, 1, 3, 2, 4)  # (bsz, pixel, heads, len_t, dim // heads)

        q = q * self.scale
        attn = torch.matmul(q, k).softmax(dim=-1)
        out = torch.matmul(attn.unsqueeze(1), v)
        # b (height width) h n d -> b n (h d) height width
        out = out.permute(0, 3, 2, 4, 1).contiguous()
        out = out.view(bsz, len_s, -1, height, width)
        return out


class FullVideoAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)

    def forward(self, q, k, v):
        bsz = q.shape[0]
        len_s = q.shape[1]
        len_t = k.shape[1]
        height, width = q.shape[-2:]
        # b m c h w -> b (h w) m c
        q = q.permute(0, -1, -2, 1, 2).contiguous()
        q = q.view(bsz, height * width, len_s, -1)
        k = k.permute(0, -1, -2, 1, 2).contiguous()
        k = k.view(bsz, height * width, len_t, -1)
        v = v.permute(0, -1, -2, 1, 2).contiguous()
        v = v.view(bsz, height * width, len_t, -1)
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)

        q = q.view(bsz, height * width, len_s, self.heads, -1)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.view(bsz, height * width, len_t, self.heads, -1)
        k = k.permute(0, 1, 3, 4, 2)
        v = v.view(bsz, height * width, len_t, self.heads, -1)
        v = v.permute(0, 1, 3, 2, 4)

        q = q * self.scale
        attn = torch.matmul(q, k).softmax(dim=-1)
        out = torch.matmul(attn, v)
        # b (height width) h n d -> b n (h d) height width
        out = out.permute(0, 3, 2, 4, 1).contiguous()
        out = out.view(bsz, len_s, -1, height, width)
        return out


class PreNormConvGRU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = VideoLayerNorm(dim)
        self.f = ConvGRU(dim)

    def forward(self, x):
        x = self.ln(x)
        return self.f(x)


class PreNormChannelVideoAttention(nn.Module):
    def __init__(self, dim, heads, eps=1e-5):
        super().__init__()
        self.ln = VideoLayerNorm(dim, eps)
        self.f = ChannelVideoAttention(dim, heads)

    def forward(self, x):
        x = self.ln(x)
        return self.f(x, x, x)


class PreNormLastQueryFullVideoAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln = VideoLayerNorm(dim)
        self.f = FullVideoAttention(dim, heads)

    def forward(self, x):
        q = x[:, [-1]]
        q = self.ln(q)
        q = self.f(q, x, x)
        z = torch.zeros_like(x[:, :-1])
        return torch.cat([z, q], dim=1)


def FFN3D(dim, s=2):
    return SimpleLayer2D(FFN(dim, s))


class PreNormFFN3D(nn.Module):
    def __init__(self, dim, s=2):
        super().__init__()
        self.ln = VideoLayerNorm(dim)
        self.f = FFN3D(dim, s)

    def forward(self, x):
        x = self.ln(x)
        return self.f(x)


class PreNormTimeConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = VideoLayerNorm(dim)
        self.f = TimeConv(dim)

    def forward(self, x):
        x = self.ln(x)
        return self.f(x)


class VideoBlock(nn.Module):
    def __init__(self, dim, heads, n_layers):
        super().__init__()
        self.time_emb = TimeEmbedding(dim)
        layers = []
        for _ in range(n_layers):
            layers.extend(
                [
                    SimpleLayer2D(Block(dim)),
                    PreNormTimeConv(dim),
                    PreNormLastQueryFullVideoAttention(dim, heads),
                    PreNormFFN3D(dim),
                ]
            )
        self.layers = ReversibleSequential(layers, split_dim=2)

    def forward(self, x):
        x = self.time_emb(x)
        x = self.layers(x)
        return x
