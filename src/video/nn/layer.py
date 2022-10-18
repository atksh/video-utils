import math

import torch
from einops import rearrange
from torch import nn
from torch.jit import Final
from torch.nn import functional as F

NEG_INF = -5000.0


class VideoToImage(nn.Module):
    def forward(self, x):
        """(batch_size, seq_len, channel, height, width) -> (batch_size * seq_len, channel, height, width)"""
        return rearrange(x, "b s c h w -> (b s) c h w")


class ImageToVideo(nn.Module):
    def forward(self, x, batch_size):
        """(batch_size * seq_len, channel, height, width) -> (batch_size, seq_len, channel, height, width)"""
        return rearrange(x, "(b s) c h w -> b s c h w", b=batch_size)


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


class LayerNorm2D(nn.Module):
    eps: Final[float]

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        x = x * self.gamma + self.beta
        return x


class SELayer(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        intermed_dim = max(dim // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(dim, intermed_dim, bias=False),
            nn.Mish(),
            nn.Linear(intermed_dim, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LRASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm2D(out_channels),
            nn.Mish(),
        )
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.aspp1(x) * self.aspp2(x)


class FFN(nn.Module):
    def __init__(self, dim, s=2):
        super().__init__()
        self.wi = nn.Conv2d(dim, dim * s * 2, kernel_size=1, padding=0, bias=False)
        self.wo = nn.Conv2d(dim * s, dim, kernel_size=1, padding=0, bias=False)
        self.act = nn.Mish()

    def forward(self, x):
        x1, x2 = self.wi(x).chunk(2, dim=1)
        x = x1 * self.act(x2)
        x = self.wo(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor = None
        self.rem = None

        if in_dim < out_dim:
            self.rem = out_dim - in_dim
            self.shortcut = nn.Conv2d(in_dim, self.rem, kernel_size=1, bias=False)
        elif in_dim > out_dim:
            self.factor = math.ceil(in_dim / out_dim)  # >= 1
            self.rem = out_dim - in_dim // self.factor
            if self.rem > 0:
                self.shortcut = nn.Conv2d(in_dim, self.rem, kernel_size=1, bias=False)

    def forward(self, x):
        if self.in_dim == self.out_dim:
            pass
        elif self.in_dim < self.out_dim:
            x = torch.cat([x, self.shortcut(x)], dim=1)
        else:
            b, _, h, w = x.shape
            y = x[:, : (self.in_dim // self.factor) * self.factor]
            y = y.reshape(b, self.factor, -1, h, w).mean(dim=1)
            if self.rem > 0:
                x = torch.cat([y, self.shortcut(x)], dim=1)
            else:
                x = y
        assert x.shape[1] == self.out_dim, f"{x.shape[1]} != {self.out_dim}"
        return x


class Upsample(nn.Module):
    def __init__(self, scale=2, mode="bilinear", align_corners=True):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def interpolate(self, x, **kwargs):
        return F.interpolate(
            x, mode=self.mode, align_corners=self.align_corners, **kwargs
        )

    def forward(self, lowres, highres):
        upsampled = self.interpolate(lowres, scale_factor=self.scale)
        h1, w1 = highres.shape[-2:]
        h2, w2 = upsampled.shape[-2:]
        if h1 != h2 or w1 != w2:
            size = (h1, w1)
            upsampled = self.interpolate(upsampled, size=size)
        out = torch.cat([upsampled, highres], dim=1)
        return out


class UpsampleWithRefrence(Upsample):
    def __init__(self, low_dim, high_dim, scale=2, mode="bilinear", align_corners=True):
        super().__init__(scale, mode, align_corners)
        self.high_dim = high_dim
        self.to_ref = nn.Conv2d(
            low_dim, 2 * high_dim, kernel_size=3, padding=1, bias=False
        )
        self.up = Upsample(scale, mode, align_corners)

    def forward(self, lowres, highres):
        size = highres.shape[-2:]
        high_dim = highres.shape[-3]

        x = self.interpolate(lowres, size=size)
        b = x.shape[0]
        ref = self.to_ref(x)
        ref = ref.view(b * high_dim, 2, *size)
        highres = highres.reshape(b * high_dim, 1, *size)

        out = self.transform(ref, highres)
        out = out.view(b, high_dim, *size)
        highres = highres.view(b, high_dim, *size)
        return self.up(lowres, torch.cat([out, highres], dim=1))

    def transform(self, ref_xy, source):
        # ref_xy: (batch_size, 2, height, width)
        # source: (batch_size, dim, height, width)
        ref_x = ref_xy[:, 0].softmax(dim=-1).cumsum(dim=-1)
        ref_y = ref_xy[:, 1].softmax(dim=-2).cumsum(dim=-2)
        ref_xy = torch.stack([ref_y, ref_x], dim=-1) * 2 - 1
        out = F.grid_sample(source, ref_xy, align_corners=True)
        return out


class MBConv(nn.Module):
    def __init__(self, dim, s=4, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.p1 = nn.Conv2d(dim, dim * s, kernel_size=1, bias=False)
        self.d1 = nn.Conv2d(
            dim * s, dim * s, kernel_size=kernel_size, padding=padding, groups=dim * 4
        )
        self.se = SELayer(dim * s)
        self.p2 = nn.Conv2d(dim * s, dim, kernel_size=1, bias=False)
        self.act = nn.Mish()
        self.ln = LayerNorm2D(dim)

    def forward(self, x):
        resid = x
        x = self.p1(x)
        x = self.d1(x)
        x = self.act(x)
        x = self.se(x)
        x = self.p2(x)
        x = self.ln(x + resid)
        return x


class ImageBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, n_layers=3):
        super().__init__()
        self.lraspp = LRASPP(in_dim, out_dim)
        self.mbconvs = nn.Sequential(
            *[MBConv(out_dim, kernel_size=kernel_size) for _ in range(n_layers)]
        )
        self.sc = ShortCut(in_dim, out_dim)
        self.ln = LayerNorm2D(out_dim)

    def forward(self, x):
        resid = self.sc(x)
        x = self.ln(self.lraspp(x) + resid)
        x = self.mbconv(x)
        return x


# 3D layers


def VideoLayerNorm(dim, eps=1e-6):
    return Layer2D(LayerNorm2D(dim, eps))


class ConvGRU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_z = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.conv_r = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.conv_h = nn.Conv2d(2 * dim, dim, kernel_size=1)

    def forward(self, video):
        # video: (batch_size, seq_len, dim, height, width)
        seq_len = video.shape[1]
        out = []
        h = torch.zeros_like(video[:, 0])
        for i in range(seq_len):
            x = video[:, i]
            xh = torch.cat([x, h], dim=1)
            z = torch.sigmoid(self.conv_z(xh))
            r = torch.sigmoid(self.conv_r(xh))
            xh = torch.cat([x, r * h], dim=1)
            h = (1 - z) * h + z * torch.tanh(self.conv_h(xh))
            out.append(h)
        out = torch.stack(out, dim=1)
        return out


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
        height, width = q.shape[-2:]
        q = self.Q(q.mean(dim=[-1, -2]))  # (batch_size, len_s, dim)
        k = self.K(k.mean(dim=[-1, -2]))  # (batch_size, len_t, dim)
        v = self.V(rearrange(v, "b m c h w -> b (h w) m c"))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b m (h d) -> b h d m", h=self.heads)
        v = rearrange(v, "b hw m (h d) -> b hw h m d", h=self.heads)

        q = q * self.scale
        attn = torch.matmul(q, k).softmax(dim=-1)
        out = torch.matmul(attn.unsqueeze(1), v)
        out = rearrange(
            out,
            "b (height width) h n d -> b n (h d) height width",
            h=self.heads,
            height=height,
            width=width,
        )
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
        height, width = q.shape[-2:]
        q = self.Q(rearrange(q, "b m c h w -> b (h w) m c"))
        k = self.K(rearrange(k, "b m c h w -> b (h w) m c"))
        v = self.V(rearrange(v, "b m c h w -> b (h w) m c"))

        q = rearrange(q, "b hw n (h d) -> b hw h n d", h=self.heads)
        k = rearrange(k, "b hw m (h d) -> b hw h d m", h=self.heads)
        v = rearrange(v, "b hw m (h d) -> b hw h m d", h=self.heads)

        q = q * self.scale
        attn = torch.matmul(q, k).softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(
            out,
            "b (height width) h n d -> b n (h d) height width",
            h=self.heads,
            height=height,
            width=width,
        )
        return out


def FFN3D(dim, s=2):
    return Layer2D(FFN(dim, s))


def ImageBlock3D(in_dim, out_dim, kernel_size=7):
    return Layer2D(ImageBlock(in_dim, out_dim, kernel_size))


class VideoBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.image = ImageBlock3D(dim, dim)
        self.conv_gru = ConvGRU(dim)
        self.channel_attn = ChannelVideoAttention(dim, heads)
        self.full_attn = FullVideoAttention(dim, heads)
        self.ffn = FFN3D(dim)
        self.ln1 = VideoLayerNorm(dim)
        self.ln2 = VideoLayerNorm(dim)
        self.ln3 = VideoLayerNorm(dim)
        self.ln4 = VideoLayerNorm(dim)

    def last_only_forward(self, f, ln, x):
        q = x[:, [-1]]
        q = ln(f(q) + q)
        x = torch.cat([x[:, :-1], q], dim=1)
        return x

    def forward(self, x):
        # x: (batch_size, len, dim, height, width)
        x = self.image(x)
        resid = x
        x = self.ln1(x + self.conv_gru(x))
        x = self.ln2(x + self.channel_attn(x, x, x))
        full_attn = lambda q: self.full_attn(q, x, x)
        x = self.last_only_forward(full_attn, self.ln3, x)
        x = self.ln4(x + self.ffn(x) + resid)
        return x
