import torch
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


def ckpt_einsum(equation, *args):
    def einsum_func(*args):
        return torch.einsum(equation, *args)

    return ckpt_forward(einsum_func)(*args)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class VideoToImage(nn.Module):
    def forward(self, x):
        """(batch_size, seq_len, channel, height, width) -> (batch_size * seq_len, channel, height, width)"""
        return rearrange(x, "b s c h w -> (b s) c h w")


class ImageToVideo(nn.Module):
    def forward(self, x, batch_size):
        """(batch_size * seq_len, channel, height, width) -> (batch_size, seq_len, channel, height, width)"""
        return rearrange(x, "(b s) c h w -> b s c h w", b=batch_size)


class ToChannelLastImage(nn.Module):
    def forward(self, x):
        """(batch_size, channel, height, width) -> (batch_size, height, width, channel)"""
        return rearrange(x, "b c h w -> b h w c")


class ToChannelFirstImage(nn.Module):
    def forward(self, x):
        """(batch_size, height, width, channel) -> (batch_size, channel, height, width)"""
        return rearrange(x, "b h w c -> b c h w")


class ToChannelLastVideo(nn.Module):
    def forward(self, x):
        """(batch_size, seq_len, channel, height, width) -> (batch_size, seq_len, height, width, channel)"""
        return rearrange(x, "b s c h w -> b s h w c")


class ToChannelFirstVideo(nn.Module):
    def forward(self, x):
        """(batch_size, seq_len, height, width, channel) -> (batch_size, seq_len, channel, height, width)"""
        return rearrange(x, "b s h w c -> b s c h w")


class VideoLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    @ckpt_forward
    def forward(self, x):
        # (batch_size, seq_len, channel, height, width)
        assert is_video(x)
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True)
        normed = (x - mu) / (std + self.eps)
        gamma = self.gamma.view(1, 1, -1, 1, 1)
        beta = self.beta.view(1, 1, -1, 1, 1)
        return normed * gamma + beta


class Conv2d(nn.Module):
    def __init__(self, dim, out_dim, kernel_size, padding, groups):
        super().__init__()
        dim1 = dim // 2
        dim2 = dim - dim1
        assert dim % groups == 0

        self.dim1 = dim1
        self.dim2 = dim2
        self.heightwise = nn.Conv2d(
            dim1, dim1, (kernel_size, 1), padding=(padding, 0), bias=False, groups=dim1
        )
        self.widthwise = nn.Conv2d(
            dim2, dim2, (1, kernel_size), padding=(0, padding), bias=False, groups=dim2
        )
        self.pointwise1 = nn.Conv2d(dim, dim, 1, bias=False, groups=groups)
        self.depthwise = nn.Conv2d(dim, dim, 3, padding=1, bias=False, groups=dim)
        self.pointwise2 = nn.Conv2d(dim, dim, 1, bias=False, groups=groups)
        if dim != out_dim:
            self.out = nn.Conv2d(dim, out_dim, 1, bias=False)
        else:
            self.out = nn.Identity()

    @ckpt_forward
    def forward(self, x):
        assert is_image(x)
        x = self.pointwise1(x)
        x1 = x[:, : self.dim1]
        x2 = x[:, self.dim1 :]
        x1 = self.heightwise(x1)
        x2 = self.widthwise(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.pointwise2(x)
        x = self.depthwise(x)
        x = self.out(x)
        return x


class SELayer(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        intermed_dim = max(dim // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(dim, intermed_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(intermed_dim, dim, bias=False),
            nn.Sigmoid(),
        )

    @ckpt_forward
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


class GroupLinear(nn.Module):
    def __init__(self, in_dim, out_dim, groups):
        super().__init__()
        assert in_dim % groups == 0
        assert out_dim % groups == 0
        self.groups = groups
        std = (2 / (in_dim + out_dim)) ** 0.5
        self.W = nn.Parameter(
            torch.randn(groups, out_dim // groups, in_dim // groups) * std
        )

    @ckpt_forward
    def forward(self, x):
        x = x.reshape(x.shape[:-1] + (self.groups, -1))
        x = ckpt_einsum("...gj,...gij->...gi", x, self.W)
        return x.reshape(x.shape[:-2] + (-1,))


class ChannelVideoAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.Q = GroupLinear(dim, dim, heads)
        self.K = GroupLinear(dim, dim, heads)
        self.V = GroupLinear(dim, dim, heads)
        self.softmax = SoftmaxDropout(p=0.1, dim=-1)

    @ckpt_forward
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
        attn = ckpt_einsum("bhnd,bhmd->bhnm", q, k)
        casual_mask = (
            torch.triu(torch.ones(attn.shape[-2:]), diagonal=1).bool().to(attn.device)
        )
        attn = attn.masked_fill(casual_mask, NEG_INF)
        attn = self.softmax(attn)
        out = ckpt_einsum("bhnm,bshmd->bshnd", attn, v)
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

        self.Q = GroupLinear(dim, dim, heads)
        self.K = GroupLinear(dim, dim, heads)
        self.V = GroupLinear(dim, dim, heads)
        self.softmax = SoftmaxDropout(p=0.1, dim=-1)

    @ckpt_forward
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
        attn = ckpt_einsum("bshnd,bshmd->bshnm", q, k)
        casual_mask = (
            torch.triu(torch.ones(attn.shape[-2:]), diagonal=1).bool().to(attn.device)
        )
        attn = attn.masked_fill(casual_mask, NEG_INF)
        attn = self.softmax(attn)
        out = ckpt_einsum("bshnm,bshmd->bshnd", attn, v)
        out = rearrange(
            out,
            "b (height width) h n d -> b n (h d) height width",
            h=self.heads,
            height=height,
            width=width,
        )
        return out


class FFN(nn.Module):
    def __init__(self, dim, groups, s=2, kernel_size=7):
        super().__init__()
        assert dim % groups == 0
        padding = (kernel_size - 1) // 2
        self.wi1 = Conv2d(
            dim,
            dim * s,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.wi2 = Conv2d(
            dim,
            dim * s,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.wo = Conv2d(
            dim * s,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    @ckpt_forward
    def forward(self, x):
        assert is_video(x)
        b = x.shape[0]
        x = self.to_image(x)

        x1 = self.wi1(x)
        x2 = self.wi2(x)
        x = x1 * F.gelu(x2)

        x = self.wo(x)
        x = self.to_video(x, b)
        return x


class Layer2D(nn.Module):
    def __init__(self, in_dim, out_dim, groups):
        super().__init__()
        self.conv = Conv2d(in_dim, in_dim, 7, 3, groups)
        self.act = nn.GELU()
        self.se = SELayer(in_dim)
        self.post_ln = VideoLayerNorm(out_dim)
        if in_dim != out_dim:
            self.skip = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        else:
            self.skip = nn.Identity()
        self.mix = Conv2d(in_dim, out_dim, 7, 3, 1)

        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    @ckpt_forward
    def forward(self, x):
        assert is_video(x)
        bsz = x.shape[0]
        resid = self.to_video(self.skip(self.to_image(x)), bsz)
        x = self.to_image(x)

        x = self.conv(x)
        x = self.act(x)
        x = self.se(x)
        x = self.mix(x)

        x = self.to_video(x, bsz)
        x = self.post_ln(x + resid)
        return x


class Layer3D(nn.Module):
    def __init__(self, dim, heads, without_channel_attention=False):
        super().__init__()
        self.without_channel_attention = without_channel_attention
        if without_channel_attention:
            self.channel_attn = None
            self.ln1 = None
        else:
            self.channel_attn = ChannelVideoAttention(dim, heads)
            self.ln1 = VideoLayerNorm(dim)

        self.full_attn = FullVideoAttention(dim, heads)
        self.ffn = FFN(dim, heads)
        self.post = Layer2D(dim, dim, heads)

        self.ln2 = VideoLayerNorm(dim)
        self.ln3 = VideoLayerNorm(dim)

    def last_only_forward(self, f, ln, x):
        q = x[:, [-1]]
        q = ln(q + f(q))
        x = torch.cat([x[:, :-1], q], dim=1)
        return x

    @ckpt_forward
    def forward(self, x):
        # x: (batch_size, len, dim, height, width)
        assert is_video(x)
        resid = x
        if not self.without_channel_attention:
            x = self.ln1(x + self.channel_attn(x, x, x))

        full_attn = lambda q: self.full_attn(q, x, x)
        x = self.last_only_forward(full_attn, self.ln2, x)
        x = self.ln3(x + self.ffn(x) + resid)
        x = self.post(x)
        return x


class Upsample(nn.Module):
    def __init__(self, scale=2, mode="bicubic", align_corners=True):
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


if __name__ == "__main__":
    m = Layer3D(32, 2, 4)
    h, w = (320, 480)
    x = torch.randn(1, 10, 32, h, w)
    # y = m(x[:, -5:], x, x)
    y = m(x)
    print(y.shape)
