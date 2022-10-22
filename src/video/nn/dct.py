from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.jit import Final
from torchjpeg.dct import block_dct, block_idct, blockify, deblockify

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
            module (nn.Module): Module to apply to DCT coefficients with the shape of (bsz, n_blocks, ch, block_size ** 2)
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
        return x.gather(-1, idx).contiguous()

    def from_zigzag(self, x):
        bsz, ch, n = x.shape[:3]
        x = x.view(bsz, ch, n, self.block_size**2)
        inv_idx = torch.LongTensor(self.inv_idx, device=x.device).expand_as(x)
        x = x.gather(-1, inv_idx).contiguous()
        return x.view(bsz, ch, n, self.block_size, self.block_size)

    def pre(self, x):
        bsz, in_ch = x.shape[:2]
        x = blockify(x, self.block_size)
        x = aot_block_dct(x)
        if self.zigzag:
            x = self.to_zigzag(x)
        else:
            x = x.reshape(bsz, in_ch, -1, self.block_size**2)
        x = x.contiguous()  # (bsz, in_ch, n_blocks, block_size ** 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def post(self, x, size):
        x = x.permute(0, 2, 1, 3).contiguous()
        bsz, out_ch = x.shape[:2]
        if self.zigzag:
            x = self.from_zigzag(x)
        else:
            x = x.reshape(bsz, out_ch, -1, self.block_size, self.block_size)
        x = aot_block_idct(x)
        x = deblockify(x, size).contiguous()
        return x

    def forward(self, x):
        size = x.shape[-2:]
        if min(*size) < self.block_size:
            raise ValueError(
                "Image size must be at least "
                f"{self.block_size}x{self.block_size}"
                f" but got {size[0]}x{size[1]}"
            )
        # (b, c_in, h, w) -> (b, n_blocks, c_in, block_size ** 2)
        x = self.pre(x)
        # (b, n_blocks, c_in, block_size ** 2) -> (b, n_blocks, c_out, block_size ** 2)
        x = self.module(x)
        # (b, n_blocks, c_out, block_size ** 2) -> (b, c_out, h, w)
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


class FreqConv(nn.Module):
    def __init__(self, in_ch, out_ch, window_size, groups=1):
        super().__init__()
        self.padding = window_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, window_size, padding=0, groups=groups)

    def forward(self, x):
        bsz, n, ch, dim = x.shape
        x = x.view(bsz * n, ch, dim)
        x = F.pad(x, (0, self.padding))
        x = self.conv(x)
        x = x.view(bsz, n, -1, x.shape[-1])
        x = x.contiguous()
        return x


class DCTFreqConv(nn.Module):
    def __init__(self, in_ch, out_ch, window_size: int, block_size: int, groups=1):
        super().__init__()
        conv = FreqConv(in_ch, out_ch, window_size=window_size, groups=groups)
        self.sand = BlockDCTSandwich(conv, block_size, zigzag=True)

    def forward(self, x):
        return self.sand(x)


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
        dims = [max(64, out_dim // 2), max(128, out_dim), max(256, out_dim * 2)]
        self.mlp = nn.Sequential(
            nn.LazyLinear(dims[0]),
            nn.SiLU(),
            nn.Linear(dims[0], dims[1]),
            nn.SiLU(),
            nn.Linear(dims[1], dims[2]),
            nn.SiLU(),
            nn.Linear(dims[2], out_dim),
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
        x = self.x.view(1, -1, 1)
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
        x = self.mlp(x)  # (1, n, out_dim)
        x = self.soft_clip(x)
        return x.view(-1, n, *self.shape)


class PassFilter(nn.Module):
    def __init__(self, ch: int, block_size: int):
        super().__init__()
        length = block_size**2
        self.x = nn.Parameter(torch.zeros(ch, length))

    def forward(self, inp):
        # x: (b, n_blocks, ch, block_size**2)
        assert self.x.shape[0] == inp.shape[-2]
        assert self.x.shape[1] == inp.shape[-1]

        s1 = F.softmax(self.x, dim=-1).cumsum(dim=-1)
        s2 = torch.flip(self.x, dim=-1).softmax(dim=-1).cumsum(dim=-1).flip(dim=-1)
        s = torch.minimum(s1, s2)  # (ch, length)
        s = s.unsqueeze(0).unsqueeze(0)  # (1, 1, ch, length)
        return inp * s
