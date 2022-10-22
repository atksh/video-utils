import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.jit import Final
from torchjpeg.dct import block_dct, block_idct, blockify, deblockify


class BlockDCTSandwich(nn.Module):
    idx: Final[List[int]]
    inv_idx: Final[List[int]]
    block_size: Final[int]
    zigzag: Final[bool]

    def __init__(self, module: nn.Module, block_size: int, zigzag: bool = False):
        """DCT -> module -> IDCT

        Args:
            module (nn.Module): Module to apply to DCT coefficients with the shape of (bsz, block_size**2, ch, h', w')
            block_size (int): Size of the block to use for DCT
        """
        super().__init__()
        self.module = module
        self.block_size = block_size
        self.zigzag = zigzag
        if self.zigzag:
            self.idx, self.inv_idx = self.build_idx()
        else:
            self.idx = list(range(self.block_size**2))
            self.inv_idx = list(range(self.block_size**2))

    def build_idx(self):
        b = self.block_size

        def to_key(x):
            s = x[0] + x[1]
            o = b**2 * s
            if s % 2 == 1:
                o += x[0]
            else:
                o -= x[0]
            return o

        # out = [[0, 0], [0, 1], [1, 0], [2, 0], [1, 1], [0, 2], ...]
        out = [[i, j] for i in range(b) for j in range(b)]
        out = list(sorted(out, key=to_key))
        out = torch.tensor(out).view(b, b, 2)
        out = torch.arange(b) * out[..., 0] + out[..., 1]
        inv_out = torch.argsort(out)
        return out.tolist(), inv_out.tolist()

    def to_zigzag(self, x):
        bsz, ch, n = x.shape[:3]
        x = x.view(bsz, ch, n, self.block_size**2)
        idx = torch.tensor(self.idx, device=x.device, dtype=torch.long)
        idx = idx.view(1, 1, 1, -1).expand_as(x)
        return x.gather(-1, idx)

    def from_zigzag(self, x):
        bsz, ch, n = x.shape[:3]
        x = x.view(bsz, ch, n, self.block_size**2)
        inv_idx = torch.tensor(self.inv_idx, device=x.device, dtype=torch.long)
        inv_idx = inv_idx.view(1, 1, 1, -1).expand_as(x)
        x = x.gather(-1, inv_idx)
        return x.view(bsz, ch, n, self.block_size, self.block_size)

    def out_size(self, h, w):
        out_h = math.ceil((h - self.block_size + 1) / self.block_size)
        out_w = math.ceil((w - self.block_size + 1) / self.block_size)
        return out_h, out_w

    def pre(self, x):
        bsz, in_ch = x.shape[:2]
        h, w = x.shape[-2:]
        x = blockify(x, self.block_size)
        x = block_dct(x)
        if self.zigzag:
            x = self.to_zigzag(x)
        else:
            x = x.reshape(bsz, in_ch, -1, self.block_size**2)
        x = x.permute(0, 2, 1, 3)  # (bsz, n_blocks, in_ch, block_size ** 2)
        out_h, out_w = self.out_size(h, w)
        x = x.reshape(bsz, out_h, out_w, in_ch, self.block_size**2)
        x = x.permute(0, 4, 3, 1, 2)
        return x

    def post(self, x, size):
        bsz, _, out_ch, out_h, out_w = x.shape
        x = x.view(bsz, -1, out_ch, out_h * out_w)
        x = x.permute(0, 2, 3, 1)
        if self.zigzag:
            x = self.from_zigzag(x)
        else:
            x = x.reshape(bsz, out_ch, -1, self.block_size, self.block_size)
        x = block_idct(x)
        x = deblockify(x, size)
        return x

    def forward(self, x):
        size = x.shape[-2:]
        if min(*size) < self.block_size:
            raise ValueError(
                "Image size must be at least "
                f"{self.block_size}x{self.block_size}"
                f" but got {size[0]}x{size[1]}"
            )
        x = self.pre(x)
        x = self.module(x)
        x = self.post(x, size)
        return x


class DCT(BlockDCTSandwich):
    def __init__(self, block_size: int, zigzag: bool = False):
        super().__init__(nn.Identity(), block_size, zigzag)

    def forward(self, x):
        return super().pre(x)


class IDCT(BlockDCTSandwich):
    def __init__(self, block_size: int, zigzag: bool = False):
        super().__init__(nn.Identity(), block_size, zigzag)

    def forward(self, x, size):
        return super().post(x, size)


class DeepSpace(nn.Module):
    lb: Optional[float]
    ub: Optional[float]
    shape: Tuple[int]

    def __init__(
        self,
        n: int,
        shape: Union[int, Tuple[int], List[int]],
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ):
        super().__init__()
        self.lb = lb
        self.ub = ub
        if isinstance(shape, int):
            shape = [shape]
        self.shape = tuple(shape)
        out_dim = np.prod(self.shape)

        self._x = torch.linspace(0, 1, n).tolist()
        hidden_dim = max(32, int(np.sqrt(n * out_dim)) // 2)
        self.mlp = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    @property
    def x(self):
        return torch.tensor(self._x, device=self.mlp[0].weight.device)

    def soft_clip(self, x):
        if self.lb is None and self.ub is None:
            return x
        elif self.lb is None and self.ub is not None:
            ub = torch.tensor(self.ub, device=x.device)
            _x = torch.clamp(x, max=ub)
            return _x.detach() + x - x.detach()
        elif self.lb is not None and self.ub is None:
            lb = torch.tensor(self.lb, device=x.device)
            _x = torch.clamp(x, min=lb)
            return _x.detach() + x - x.detach()
        else:
            lb = torch.tensor(self.lb, device=x.device)
            ub = torch.tensor(self.ub, device=x.device)
            _x = torch.clamp(x, min=lb, max=ub)
            return _x.detach() + x - x.detach()

    def forward(self):
        x = self.x.view(-1, 1)
        xs = []
        for i in range(4):
            xs.append(x.pow(i + 1))
            xs.append(torch.sin(x * (i + 1) * 3.1415))
            xs.append(torch.cos(x * (i + 1) * 3.1415))
        xs.append(torch.log(x + 1e-4))
        xs.append(torch.exp(x))
        x = torch.cat(xs, dim=-1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.std(dim=1, keepdim=True)
        x = self.mlp(x)  # (n, out_dim)
        x = self.soft_clip(x)
        return x.view(x.shape[0], *self.shape)


class FreqPassFilter(nn.Module):
    def __init__(self, ch: int, block_size: int):
        super().__init__()
        length = block_size**2
        self.x = nn.Parameter(torch.zeros(length, ch))

    def reverse_order(self, x, dim: int):
        idx = torch.arange(x.shape[dim] - 1, -1, -1, device=x.device)
        return x.index_select(dim, idx)

    def get_filter(self):
        s1 = F.softmax(self.x, dim=-1).cumsum(dim=-1)
        s2 = self.reverse_order(self.x, dim=-1).softmax(dim=-1).cumsum(dim=-1)
        s = torch.minimum(s1, self.reverse_order(s2, dim=-1))  # (ch, length)
        s = s / s.sum(dim=-1, keepdim=True)  # normalize
        return s

    def forward(self, inp):
        # x: (b, block_size**2, ch, h, w)
        s = self.get_filter()
        s = s[None, :, :, None, None]
        return inp * s


class FreqCondFFN(nn.Module):
    """FFN conditioned on frequency"""

    def __init__(self, dim: int, block_size: int, expand_factor: int = 2):
        super().__init__()
        shape = (3, dim * expand_factor, dim)
        self.params = DeepSpace(n=block_size**2, shape=shape)

    def get_weights(self):
        w1, w2, w3 = self.params().unbind(dim=1)
        w3 = w3.transpose(-1, -2)
        return w1, w2, w3

    def forward(self, x):
        # x: (b, block_size**2, ch, h, w)
        # to last channel
        x = x.permute(0, 1, 3, 4, 2)
        w1, w2, w3 = self.get_weights()  # (block_size**2, dim * expand_factor, dim)
        x1 = torch.einsum("blhwc,ldc->blhwd", x, w1)
        x2 = torch.einsum("blhwc,ldc->blhwd", x, w2)
        x = x1 * F.silu(x2)
        x = torch.einsum("blhwc,ldc->blhwd", x, w3)
        x = x.permute(0, 1, 4, 2, 3)
        return x


class FreqCondConv2dBase(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, block_size):
        super().__init__()
        self.block_size = block_size
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        shape = (out_ch, in_ch * kernel_size**2)
        self.params = DeepSpace(n=block_size**2, shape=shape)

    def im2col(self, x):
        # x: (b, block_size**2, ch, h, w)
        bsz, n, ch, h, w = x.shape
        x = x.reshape(-1, ch, h, w)  # (b * n, ch, h, w)
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        # (b * n, ch * kernel_size**2, L)
        return x.view(bsz, n, *x.shape[1:])

    def col2im(self, x, size):
        # x: (b, n, ch, L) where L = h * w
        bsz, n = x.shape[:2]
        h, w = size
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        x = x.view(bsz, n, -1, out_h, out_w)
        return x

    def forward(self, x):
        # x: (b, block_size**2, ch, h, w)
        size = x.shape[-2:]
        x = self.im2col(x)  # (b, n, ch * kernel_size**2, L)
        w = self.params()  # (n, out_ch, in_ch * kernel_size**2)
        x = x.permute(0, 1, 3, 2)  # (b, n, L, ch * kernel_size**2)
        x = torch.einsum("bnlc,ndc->bnld", x, w)  # (b, n, L, out_ch)
        x = x.permute(0, 1, 3, 2)  # (b, n, out_ch, L)
        x = self.col2im(x.contiguous(), size)
        return x


class GroupFreqCondConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups, block_size):
        super().__init__()
        if in_ch % groups != 0:
            raise ValueError(f"in_ch ({in_ch}) must be divisible by groups ({groups})")
        if out_ch % groups != 0:
            raise ValueError(
                f"out_ch ({out_ch}) must be divisible by groups ({groups})"
            )

        self.groups = groups
        convs = []
        for _ in range(groups):
            convs.append(
                FreqCondConv2dBase(
                    in_ch // groups,
                    out_ch // groups,
                    kernel_size,
                    stride,
                    padding,
                    block_size,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        # x: (b, block_size**2, ch, h, w)
        x = [conv(x[:, :, [i]]) for i, conv in enumerate(self.convs)]
        x = torch.cat(x, dim=2)
        return x


def FreqCondConv2d(in_ch, out_ch, *, kernel_size, stride, padding, groups, block_size):
    if groups == 1:
        return FreqCondConv2dBase(
            in_ch, out_ch, kernel_size, stride, padding, block_size
        )
    else:
        return GroupFreqCondConv2d(
            in_ch, out_ch, kernel_size, stride, padding, groups, block_size
        )


class FreqCondLayerNorm(nn.Module):
    def __init__(self, ch, block_size, eps=1e-5):
        super().__init__()
        self.block_size = block_size
        self.eps = eps
        self.params = DeepSpace(n=block_size**2, shape=(2, ch))

    def forward(self, x):
        # x: (b, block_size**2, ch, h, w)
        # to last channel
        x = x.permute(0, 1, 3, 4, 2)  # (b, block_size**2, h, w, ch)
        n = x.shape[1]

        params = self.params()
        gamma, beta = params.chunk(2, dim=1)  # (n, ch)

        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        gamma = gamma.reshape(1, n, 1, 1, -1)
        beta = beta.reshape(1, n, 1, 1, -1)

        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * gamma + beta
        x = x.permute(0, 1, 4, 2, 3)  # (b, block_size**2, ch, h, w)
        return x


class FreqCondChannelLinear(nn.Module):
    def __init__(self, in_ch, out_ch, block_size):
        super().__init__()
        self.block_size = block_size
        self.params = DeepSpace(n=block_size**2, shape=(out_ch, in_ch))

    def forward(self, x):
        # x: (b, block_size**2, ch, h, w)
        # to last channel
        x = x.permute(0, 1, 3, 4, 2)  # (b, block_size**2, h, w, ch)
        w = self.params()  # (n, out_ch, in_ch)
        x = torch.einsum("bnhwc,ndc->bnhwd", x, w)
        x = x.permute(0, 1, 4, 2, 3)  # (b, block_size**2, out_ch, h, w)
        return x


class FreqCondBlock(nn.Module):
    def __init__(self, ch, block_size, kernel_size=3, expand_factor=2):
        super().__init__()
        self.conv = FreqCondConv2d(
            ch,
            ch,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=ch,
            block_size=block_size,
        )
        self.filter = FreqPassFilter(ch, block_size)
        self.ln = FreqCondLayerNorm(ch, block_size)
        self.ffn = FreqCondFFN(ch, block_size, expand_factor=expand_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.filter(x)
        x = self.ln(x)
        x = self.ffn(x)
        return x


class FreqCondStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, block_size, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(FreqCondBlock(in_ch, block_size, **kwargs))
        self.blocks = nn.ModuleList(blocks)

        self.downsample = FreqCondConv2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=1,
            block_size=block_size,
        )
        self.ln = FreqCondLayerNorm(out_ch, block_size)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        x = self.downsample(x)
        x = self.ln(x)
        return x


class FreqBackbone(nn.Module):
    def __init__(
        self, *, in_ch=3, depths=[1, 1, 1, 1], widths=[8, 16, 24, 32], block_size=8
    ):
        super().__init__()
        self.to_freq = DCT(block_size, zigzag=True)
        self.from_freq = IDCT(block_size, zigzag=True)
        self.stem = FreqCondConv2d(
            in_ch,
            widths[0],
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            block_size=block_size,
        )

        stages = []
        for i in range(len(depths)):
            in_width = widths[max(0, i - 1)]
            out_width = widths[i]
            depth = depths[i]
            stages.append(
                FreqCondStage(
                    in_ch=in_width, out_ch=out_width, depth=depth, block_size=block_size
                )
            )

        self.stages = nn.ModuleList(stages)

    def forward(self, x, return_freq=True):
        # x: (b, ch, h, w)
        h, w = x.shape[-2:]
        x = self.to_freq(x)
        x = self.stem(x)
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)

            # shape info
            # freq: (bsz, block_size**2, dim, h', w')
            # non_freq: (bsz, dim, h', w')
            if return_freq:
                feats.append(x)
            else:
                size = (h // 2 ** (i + 1), w // 2 ** (i + 1))
                feats.append(self.from_freq(x, size=size))
        return feats


class FreqHead(nn.Module):
    def __init__(self, *, in_ch=32, out_ch=1, block_size=8):
        super().__init__()
        self.proj = FreqCondChannelLinear(in_ch, out_ch, block_size)
        self.freq_proj = nn.Linear(block_size**2, 1)

    def forward(self, x):
        # x: (b, block_size**2, ch, h', w')
        x = x.mean(dim=[-1, -2], keepdim=True)  # (b, block_size**2, ch, 1, 1)
        x = self.proj(x).squeeze(-1).squeeze(-1)  # (b, block_size**2, out_ch)
        x = x.transpose(1, 2)  # (b, out_ch, block_size**2)
        x = self.freq_proj(x)  # (b, out_ch, 1)
        x = x.squeeze(-1)  # (b, out_ch)
        return x
