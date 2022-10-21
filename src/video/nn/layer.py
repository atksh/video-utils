import revlib
import torch
from torch import nn
from torch.nn import functional as F

from .dct import DCTFreqConv

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


class ReversibleSequential(nn.Module):
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


class VideoToImage(nn.Module):
    def forward(self, x):
        _, _, c, h, w = x.shape
        return x.view(-1, c, h, w)


class ImageToVideo(nn.Module):
    def forward(self, x, batch_size: int):
        _, c, h, w = x.shape
        return x.view(batch_size, -1, c, h, w)


class Upsample(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=True
        )
        return x


class DualScaleDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_hr = nn.Conv2d(3, 1, 1, bias=False)
        self.conv_lr = nn.Conv2d(3, 3, 1, bias=False)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, x):
        hr_x = self.conv_hr(x)
        lr_x = self.down(self.conv_lr(x))
        return hr_x, lr_x


class DualScaleUpsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = Upsample(scale=2)
        self.conv = nn.Conv2d(1, 3, 1, bias=False)
        self.mlp = nn.Sequential(
            DCTFreqConv(3, 12, 3, 8, 1),
            NonLinear(),
            DCTFreqConv(12, 3, 3, 8, 1),
        )

    def forward(self, hr_x, lr_x):
        x = self.conv(hr_x) + self.up(lr_x)
        x = self.mlp(x)
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


class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class SELayer(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        intermed_dim = max(dim // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(dim, intermed_dim, bias=False),
            NonLinear(),
            nn.Linear(intermed_dim, dim, bias=False),
            Sigmoid(),
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
            NonLinear(),
        )
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            Sigmoid(),
        )

    def forward(self, x):
        return self.aspp1(x) * self.aspp2(x)


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


class UpsampleWithRefrence(nn.Module):
    def __init__(self, low_dim, high_dim, scale=2, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.to_ref = nn.Conv2d(
            low_dim, 2 * high_dim, kernel_size=3, padding=1, bias=False
        )
        self.up = Upsample(scale)

    def interpolate(self, x, **kwargs):
        x = F.interpolate(x, mode="nearest", **kwargs)
        return x

    def merge(self, lowres, highres):
        upsampled = self.up(lowres)
        h1, w1 = highres.shape[-2:]
        h2, w2 = upsampled.shape[-2:]
        if h1 != h2 or w1 != w2:
            size = (h1, w1)
            upsampled = self.interpolate(upsampled, size=size)
        out = torch.cat([upsampled, highres], dim=1)
        return out

    def forward(self, lowres, highres):
        size = highres.shape[-2:]
        high_dim = highres.shape[-3]

        b = lowres.shape[0]
        ref = self.up(self.to_ref(lowres))
        h, w = ref.shape[-2:]
        if h != size[0] or w != size[1]:
            ref = self.interpolate(ref, size=size)
        ref = ref.view(b * high_dim, 2, *size)
        highres = highres.view(b * high_dim, 1, *size)

        out = self.transform(ref, highres)
        out = out.view(b, high_dim, *size)
        highres = highres.view(b, high_dim, *size)
        return self.merge(lowres, torch.cat([out, highres], dim=1))

    def transform(self, ref_xy, source):
        # ref_xy: (batch_size, 2, height, width)
        # source: (batch_size, dim, height, width)
        ref_x = ref_xy[:, 0].softmax(dim=-1).cumsum(dim=-1)
        ref_y = ref_xy[:, 1].softmax(dim=-2).cumsum(dim=-2)
        ref_xy = torch.stack([ref_y, ref_x], dim=-1) * 2 - 1
        out = F.grid_sample(source, ref_xy, mode=self.mode, align_corners=True)
        return out


class MBConv(nn.Module):
    def __init__(self, dim, s=4):
        super().__init__()
        self.p1 = nn.Conv2d(dim, dim * s, kernel_size=1, bias=False)
        self.d1 = nn.Conv2d(dim * s, dim * s, kernel_size=3, padding=1, groups=dim * s)
        self.se = SELayer(dim * s)
        self.p2 = nn.Conv2d(dim * s, dim, kernel_size=1, bias=False)
        self.act = NonLinear()
        self.ln1 = LayerNorm2D(dim)
        self.ln2 = LayerNorm2D(dim * s)

    def forward(self, x):
        x = self.ln1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.act(x)
        x = self.ln2(x)
        x = self.se(x)
        x = self.p2(x)
        return x


class ImageReduction(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers):
        super().__init__()
        self.skip = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.lraspp = LRASPP(in_dim, out_dim)
        layers = [MBConv(out_dim) for _ in range(n_layers)]
        self.layers = ReversibleSequential(layers, split_dim=1)
        self.ln = LayerNorm2D(out_dim)

    def forward(self, x):
        resid = self.skip(x)
        x = self.lraspp(x)
        x = self.layers(x)
        x = self.ln(x + resid)
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


def MBConv3D(dim, s=4):
    return SimpleLayer2D(MBConv(dim, s))


def ImageReduction3D(in_dim, out_dim, n_layers):
    return SimpleLayer2D(ImageReduction(in_dim, out_dim, n_layers))


class VideoBlock(nn.Module):
    def __init__(self, dim, heads, n_layers):
        super().__init__()
        self.time_emb = TimeEmbedding(dim)
        layers = []
        for _ in range(n_layers):
            layers.extend(
                [
                    MBConv3D(dim),
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
