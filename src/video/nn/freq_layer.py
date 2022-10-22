import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.jit import Final
from torchjpeg.dct import block_dct, block_idct, blockify, deblockify

from .einsum import Einsum
from .fuse import aot_fuse

aot_block_dct = aot_fuse(block_dct)
aot_block_idct = aot_fuse(block_idct)


class BlockDCTSandwich(nn.Module):
    idx: Final[List[int]]
    inv_idx: Final[List[int]]
    block_size: Final[int]
    zigzag: Final[bool]

    def __init__(self, module: nn.Module, block_size: int, zigzag: bool = False):
        """DCT -> module -> IDCT

        Args:
            module (nn.Module): Module to apply to DCT coefficients with the shape of (bsz, h', w', ch, block_size ** 2)
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
        return out.to_list(), inv_out.to_list()

    def to_zigzag(self, x):
        bsz, ch, n = x.shape[:3]
        x = x.view(bsz, ch, n, self.block_size**2)
        idx = torch.LongTensor(self.idx, device=x.device).expand_as(x)
        return x.gather(-1, idx)

    def from_zigzag(self, x):
        bsz, ch, n = x.shape[:3]
        x = x.view(bsz, ch, n, self.block_size**2)
        inv_idx = torch.LongTensor(self.inv_idx, device=x.device).expand_as(x)
        x = x.gather(-1, inv_idx)
        return x.view(bsz, ch, n, self.block_size, self.block_size)

    def out_size(self, h, w):
        out_h = math.ceil((h - self.block_size) / self.block_size + 1)
        out_w = math.ceil((w - self.block_size) / self.block_size + 1)
        return out_h, out_w

    def pre(self, x):
        bsz, in_ch = x.shape[:2]
        h, w = x.shape[-2:]
        x = blockify(x, self.block_size)
        x = aot_block_dct(x)
        if self.zigzag:
            x = self.to_zigzag(x)
        else:
            x = x.reshape(bsz, in_ch, -1, self.block_size**2)
        x = x.permute(0, 2, 1, 3)  # (bsz, n_blocks, in_ch, block_size ** 2)
        out_h, out_w = self.out_size(h, w)
        x = x.reshape(bsz, out_h, out_w, in_ch, self.block_size**2)
        return x

    def post(self, x, size):
        bsz, out_h, out_w, out_ch, _ = x.shape
        x = x.view(bsz, out_h * out_w, out_ch, self.block_size**2)
        x = x.permute(0, 2, 1, 3)
        if self.zigzag:
            x = self.from_zigzag(x)
        else:
            x = x.reshape(bsz, out_ch, -1, self.block_size, self.block_size)
        x = aot_block_idct(x)
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
        # (b, c_in, h, w) -> (b, h', w', c_in, block_size ** 2)
        x = self.pre(x)
        # (b, h', w', c_in, block_size ** 2) -> (b, h', w', c_out, block_size ** 2)
        x = self.module(x)
        # (b, h', w', c_out, block_size ** 2) -> (b, c_out, h, w)
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
        return super().post(x)


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

        self.x = nn.Parameter(torch.linspace(0, 1, n), requires_grad=False)
        hidden_dim = max(32, out_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

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
        self.x = nn.Parameter(torch.zeros(ch, length))

    def get_filter(self):
        s1 = F.softmax(self.x, dim=-1).cumsum(dim=-1)
        s2 = torch.flip(self.x, dim=-1).softmax(dim=-1).cumsum(dim=-1).flip(dim=-1)
        s = torch.minimum(s1, s2)  # (ch, length)
        s = s / s.sum(dim=-1, keepdim=True)  # normalize
        return s

    def forward(self, inp):
        # x: (b, h', w', ch, block_size**2)
        assert self.x.shape[0] == inp.shape[-2]
        assert self.x.shape[1] == inp.shape[-1]
        s = self.get_filter()
        s = s[None, None, None, :, :]  # (1, 1, 1, ch, length)
        return inp * s


class FreqCondFFN(nn.Module):
    """FFN conditioned on frequency"""

    def __init__(self, dim: int, block_size: int, expand_factor: int = 2):
        super().__init__()
        shape = (3, dim * expand_factor, dim)
        self.params = DeepSpace(n=block_size**2, shape=shape)
        self.einsum = Einsum("bnlc,ldc->bnld")

    def get_weights(self):
        w1, w2, w3 = self.params().unbind(dim=0)
        w3 = w3.view(w3.shape[1], -1)
        return w1, w2, w3

    def forward(self, x):
        out_h = x.shape[1]
        out_w = x.shape[2]
        x = x.view(x.shape[0], out_h * out_w, *x.shape[3:])
        # x: (b, n_blocks, ch, block_size**2)
        x = x.permute(0, 1, 3, 2)  # (b, n_blocks, block_size**2, ch)
        w1, w2, w3 = self.get_weights()  # (block_size**2, dim * expand_factor, dim)
        x1 = self.einsum(x, w1)
        x2 = self.einsum(x, w2)
        x = x1 * F.silu(x2)
        x = self.einsum(x, w3)
        x = x.permute(0, 1, 3, 2)  # (b, n_blocks, ch, block_size**2)
        x = x.reshape(x.shape[0], out_h, out_w, -1, x.shape[-1])
        return x


class FreqCondConv2dBase(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.padding = (kernel_size - 1) // 2
        self.kernel_size = kernel_size
        shape = (out_ch, in_ch * kernel_size**2)
        self.params = DeepSpace(n=block_size**2, shape=shape)

    def im2col(self, x):
        # x: (b, h, w, ch, block_size**2)
        bsz, h, w, ch, n = x.shape
        x = x.permute(0, 4, 3, 1, 2)  # (b, n, ch, h, w)
        x = x.reshape(bsz * n, ch, h, w)  # (b * n, ch, h, w)
        x = F.unfold(x, self.kernel_size, padding=self.padding)
        # (b * n, ch * kernel_size**2, L)
        return x.view(bsz, n, *x.shape[1:])

    def col2im(self, x, size):
        # x: (b, n, ch * kernel_size**2, L)
        bsz, n = x.shape[:2]
        x = x.view(bsz * n, *x.shape[2:])  # (b * n, ch * kernel_size**2, L)
        x = F.fold(x, size, self.kernel_size, padding=self.padding)  # (b * n, ch, h, w)
        x = x.reshape(bsz, -1, *x.shape[1:])  # (b, n, ch, h, w)
        x = x.permute(0, 3, 4, 2, 1)  # (b, h, w, ch, n)
        return x

    def forward(self, x):
        # x: (b, h, w, ch, block_size**2)
        size = x.shape[1:3]
        x = self.im2col(x)  # (b, n, ch * kernel_size**2, L)
        w = self.params()  # (n, out_ch, in_ch * kernel_size**2)
        x = x.permute(0, 1, 3, 2)  # (b, n, L, ch * kernel_size**2)
        x = torch.einsum("bnlc,ndc->bnld", x, w)  # (b, n, L, out_ch)
        x = x.permute(0, 1, 3, 2)  # (b, n, out_ch, L)
        x = self.col2im(x, size)  # (b, h, w, out_ch, n)
        return x


class GroupFreqCondConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, groups, block_size):
        super().__init__()
        if in_ch % groups != 0:
            raise ValueError(f"in_ch ({in_ch}) must be divisible by groups ({groups})")
        if out_ch % groups != 0:
            raise ValueError(
                f"out_ch ({out_ch}) must be divisible by groups ({groups})"
            )

        convs = []
        for _ in range(groups):
            convs.append(FreqCondConv2dBase(in_ch, out_ch, kernel_size, block_size))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        # x: (b, h, w, ch, block_size**2)
        x = torch.split(
            x, self.groups, dim=-2
        )  # (b, h, w, ch // groups, block_size**2)
        x = [conv(x_) for x_, conv in zip(x, self.convs)]
        x = torch.cat(x, dim=-2)  # (b, h, w, ch, block_size**2)
        return x


def FreqCondConv2d(in_ch, out_ch, kernel_size, groups, block_size):
    if groups == 1:
        return FreqCondConv2dBase(in_ch, out_ch, kernel_size, block_size)
    else:
        return GroupFreqCondConv2d(in_ch, out_ch, kernel_size, groups, block_size)


class FreqCondLayerNorm(nn.Module):
    def __init__(self, ch, block_size, eps=1e-5):
        super().__init__()
        self.block_size = block_size
        self.eps = eps
        self.params = DeepSpace(n=block_size**2, shape=(2, ch))

    def forward(self, x):
        # x: (b, h, w, ch, block_size**2)
        x = x.permute(0, 1, 2, 4, 3)  # (b, h, w, block_size**2, ch)
        n = x.shape[-2]
        params = self.params()
        gamma, beta = params.chunk(2, dim=1)  # (n, ch)
        mu = x.mean(dim=-1, keepdim=True)  # (b, h, w, block_size**2, 1)
        var = x.var(dim=-1, keepdim=True)  # (b, h, w, block_size**2, 1)
        gamma = gamma.reshape(1, 1, 1, n, -1)
        beta = beta.reshape(1, 1, 1, n, -1)
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * gamma + beta
        x = x.permute(0, 1, 2, 4, 3)  # (b, h, w, ch, block_size**2)
        return x


class FreqCondChannelLinear(nn.Module):
    def __init__(self, in_ch, out_ch, block_size):
        super().__init__()
        self.block_size = block_size
        self.params = DeepSpace(n=block_size**2, shape=(out_ch, in_ch))

    def forward(self, x):
        # x: (b, h, w, ch, block_size**2)
        x = x.permute(0, 1, 2, 4, 3)  # (b, h, w, block_size**2, ch)
        w = self.params()  # (n, out_ch, in_ch)
        x = torch.einsum("bhwnc,ndc->bhwnd", x, w)  # (b, h, w, block_size**2, out_ch)
        x = x.permute(0, 1, 2, 4, 3)  # (b, h, w, out_ch, block_size**2)
        return x


class FreqCondBlock(nn.Module):
    def __init__(self, ch, block_size, kernel_size=5, expand_factor=2):
        super().__init__()
        self.conv = FreqCondConv2d(ch, ch, kernel_size, ch, block_size)
        self.ln = FreqCondLayerNorm(ch, block_size)
        self.ffn = FreqCondFFN(ch, block_size, expand_factor=expand_factor)

    def forward(self, x):
        # x: (b, h, w, ch, block_size**2)
        x = self.conv(x)
        x = self.ln(x)
        x = self.ffn(x)
        return x
