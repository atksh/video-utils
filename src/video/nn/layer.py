import torch
import math
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

NEG_INF = -5000.0


def is_video(x):
    return x.ndim == 5


def is_image(x):
    return x.ndim == 4


def ckpt_forward(func):
    def wrapper(*args, **kwargs):
        all_inputs = args + tuple(kwargs.values())
        if any(torch.is_tensor(x) and x.requires_grad for x in all_inputs):
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def ckpt_seq_forward(func):
    def wrapper(*args, **kwargs):
        all_inputs = args + tuple(kwargs.values())
        batch_size = None
        for a in all_inputs:
            if torch.is_tensor(a):
                batch_size = a.shape[0]
                break
        if batch_size is not None and any(
            torch.is_tensor(x) and x.requires_grad for x in all_inputs
        ):
            new_args = []
            new_kwargs = []
            for i in range(batch_size):
                new_args.append([a[[i]] if torch.is_tensor(a) else a for a in args])
                new_kwargs.append(
                    {k: v[[i]] if torch.is_tensor(v) else v for k, v in kwargs.items()}
                )
            out = []
            for a, k in zip(new_args, new_kwargs):
                out.append(checkpoint(func, *a, **k))
            return torch.cat(out, dim=0)
        else:
            return func(*args, **kwargs)

    return wrapper


class VideoToImage(nn.Module):
    def forward(self, x):
        """(batch_size, seq_len, channel, height, width) -> (batch_size * seq_len, channel, height, width)"""
        return rearrange(x, "b s c h w -> (b s) c h w")


class ImageToVideo(nn.Module):
    def forward(self, x, batch_size):
        """(batch_size * seq_len, channel, height, width) -> (batch_size, seq_len, channel, height, width)"""
        return rearrange(x, "(b s) c h w -> b s c h w", b=batch_size)


class VideoLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # (batch_size, seq_len, channel, height, width)
        assert is_video(x)
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True)
        normed = (x - mu) / (std + self.eps)
        gamma = self.gamma.view(1, 1, -1, 1, 1)
        beta = self.beta.view(1, 1, -1, 1, 1)
        return normed * gamma + beta


class SELayer(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        intermed_dim = max(dim // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(dim, intermed_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(intermed_dim, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        assert is_image(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SoftmaxDropout(nn.Module):
    def __init__(self, p, dim):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, score):
        if self.training:
            mask = torch.empty_like(score).bernoulli_(self.p).bool()
            score = score.masked_fill(mask, NEG_INF / 2)
        score = F.softmax(score, dim=self.dim)
        return score


class ChannelVideoAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)
        self.softmax = SoftmaxDropout(p=0.1, dim=-1)

    def forward(self, q, k, v):
        assert all(is_video(x) for x in (q, k, v))
        height, width = q.shape[-2:]
        q = self.Q(q.mean(dim=[-1, -2]))  # (batch_size, len_s, dim)
        k = self.K(k.mean(dim=[-1, -2]))  # (batch_size, len_t, dim)
        v = self.V(rearrange(v, "b m c h w -> b (h w) m c"))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.heads)
        v = rearrange(v, "b hw m (h d) -> b hw h m d", h=self.heads)

        q = q * self.scale
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k)
        attn = self.softmax(attn)
        out = torch.einsum("bhnm,bshmd->bshnd", attn, v)
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
        self.softmax = SoftmaxDropout(p=0.1, dim=-1)

    def forward(self, q, k, v):
        assert all(is_video(x) for x in (q, k, v))
        height, width = q.shape[-2:]
        q = self.Q(rearrange(q, "b m c h w -> b (h w) m c"))
        k = self.K(rearrange(k, "b m c h w -> b (h w) m c"))
        v = self.V(rearrange(v, "b m c h w -> b (h w) m c"))

        q = rearrange(q, "b hw n (h d) -> b hw h n d", h=self.heads)
        k = rearrange(k, "b hw m (h d) -> b hw h m d", h=self.heads)
        v = rearrange(v, "b hw m (h d) -> b hw h m d", h=self.heads)

        q = q * self.scale
        attn = torch.einsum("bshnd,bshmd->bshnm", q, k)
        attn = self.softmax(attn)
        out = torch.einsum("bshnm,bshmd->bshnd", attn, v)
        out = rearrange(
            out,
            "b (height width) h n d -> b n (h d) height width",
            h=self.heads,
            height=height,
            width=width,
        )
        return out


class FFN(nn.Module):
    def __init__(self, dim, s=2, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.wi = nn.Conv2d(
            dim, dim * s * 2, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.wo = nn.Conv2d(
            dim * s, dim, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    def forward(self, x):
        assert is_video(x)
        b = x.shape[0]
        x = self.to_image(x)

        x = self.wi(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * F.silu(x2)

        x = self.wo(x)
        x = self.to_video(x, b)
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
        assert is_image(x)
        if self.in_dim == self.out_dim:
            pass
        elif self.in_dim < self.out_dim:
            x = torch.cat([x, self.shortcut(x)], dim=1)
        else:
            b, c, h, w = x.shape
            y = x[:, : (self.in_dim // self.factor) * self.factor]
            y = y.reshape(b, self.factor, -1, h, w).mean(dim=1)
            if self.rem > 0:
                x = torch.cat([y, self.shortcut(x)], dim=1)
            else:
                x = y
        assert x.shape[1] == self.out_dim, f"{x.shape[1]} != {self.out_dim}"
        return x


class Layer2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gn1 = nn.GroupNorm(1, in_dim)
        self.conv1 = nn.Conv2d(
            in_dim, in_dim, kernel_size=7, padding=3, bias=False, groups=in_dim
        )

        dim = max(in_dim, out_dim)
        self.conv2 = nn.Conv2d(in_dim, dim * 4, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(dim * 2, out_dim, kernel_size=1, bias=False)
        self.se = SELayer(out_dim)
        self.gn2 = nn.GroupNorm(1, out_dim)

        self.shortcut1 = ShortCut(in_dim, out_dim)
        self.shortcut2 = ShortCut(in_dim, out_dim)

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    def forward(self, x):
        assert is_video(x)
        bsz = x.shape[0]
        x = self.to_image(x)
        resid = self.shortcut1(x)

        x = self.gn1(x + self.conv1(x))
        mid = self.shortcut2(x)
        x1, x2 = self.conv2(x).chunk(2, dim=1)
        x = x1 * F.silu(x2)
        x = self.conv3(x) + mid
        x = self.se(x)
        x = self.gn2(x + resid)
        x = self.to_video(x, bsz)
        return x


class Layer3D(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.pre = Layer2D(dim, dim)
        self.channel_attn = ChannelVideoAttention(dim, heads)
        self.full_attn = FullVideoAttention(dim, heads)
        self.ffn = FFN(dim)
        self.ln1 = VideoLayerNorm(dim)
        self.ln2 = VideoLayerNorm(dim)

    def last_only_forward(self, f, x):
        q = x[:, [-1]]
        x = torch.cat([x[:, :-1], f(q)], dim=1)
        return x

    def forward(self, x):
        # x: (batch_size, len, dim, height, width)
        assert is_video(x)
        x = self.pre(x)
        resid = x
        x = self.ln1(x + self.channel_attn(x, x, x))
        full_attn = lambda q: self.full_attn(q, x, x)
        x = self.last_only_forward(full_attn, x)
        x = self.ln2(x + self.ffn(x) + resid)
        return x


class Upsample(nn.Module):
    def __init__(self, scale=2, mode="bilinear", align_corners=True):
        super().__init__()
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def interpolate(self, x, **kwargs):
        return F.interpolate(
            x, mode=self.mode, align_corners=self.align_corners, **kwargs
        )

    def forward(self, lowres, highres=None, size=None):
        assert is_video(lowres)
        assert highres is None or is_video(highres)
        bsz = lowres.shape[0]
        lowres = self.to_image(lowres)
        if size is None:
            upsampled = self.interpolate(lowres, scale_factor=self.scale)
        else:
            upsampled = lowres
        if highres is not None:
            highres = self.to_image(highres)
        h2, w2 = upsampled.shape[-2:]
        if highres is None and size is None:
            out = upsampled
        elif size is not None:
            if h2 != size[0] or w2 != size[1]:
                out = self.interpolate(upsampled, size=size)
            else:
                out = upsampled
        else:
            h1, w1 = highres.shape[-2:]
            if h1 != h2 or w1 != w2:
                size = (h1, w1)
                upsampled = self.interpolate(upsampled, size=size)
            out = torch.cat([upsampled, highres], dim=1)
        return self.to_video(out, bsz)


class UpsampleWithRefrence(Upsample):
    def __init__(self, low_dim, high_dim, scale=2, mode="bilinear", align_corners=True):
        super().__init__(scale, mode, align_corners)
        self._to_ref = nn.Sequential(
            nn.Conv2d(low_dim, low_dim * 2, kernel_size=1),
            nn.SiLU(),
            nn.GroupNorm(2, low_dim * 2),
            SELayer(low_dim * 2),
            nn.Conv2d(low_dim * 2, high_dim * 2, kernel_size=1, bias=False),
        )
        self.high_dim = high_dim

    def to_ref(self, x):
        return self._to_ref(x)

    def forward(self, lowres, highres):
        assert is_video(lowres) and is_video(highres)
        bsz = lowres.shape[0]
        size = highres.shape[-2:]
        high_dim = highres.shape[2]
        assert high_dim == self.high_dim, f"{high_dim} != {self.high_dim}"

        lowres = self.to_image(lowres)
        highres = self.to_image(highres)
        x = self.interpolate(lowres, size=size)
        b = x.shape[0]
        ref = self.to_ref(x)
        ref = ref.view(b * high_dim, 2, *size)
        highres = highres.reshape(b * high_dim, 1, *size)
        out = self.transform(ref, highres)
        out = out.view(b, high_dim, *size)
        highres = highres.view(b, high_dim, *size)
        out = torch.cat([out, highres], dim=1)
        out = self.to_video(out, bsz)
        return out

    def transform(self, ref_xy, source):
        # ref_xy: (batch_size, 2, height, width)
        # source: (batch_size, dim, height, width)
        ref_x = ref_xy[:, 0].softmax(dim=-1).cumsum(dim=-1)
        ref_y = ref_xy[:, 1].softmax(dim=-2).cumsum(dim=-2)
        ref_xy = torch.stack([ref_y, ref_x], dim=-1) * 2 - 1
        out = F.grid_sample(source, ref_xy, align_corners=True)
        return out


if __name__ == "__main__":
    m = Layer3D(32, 2, 4)
    h, w = (320, 480)
    x = torch.randn(1, 10, 32, h, w)
    # y = m(x[:, -5:], x, x)
    y = m(x)
    print(y.shape)
