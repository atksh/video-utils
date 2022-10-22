import torch
import torch.nn.functional as F
from torch import nn
from torchjpeg.dct import block_dct, block_idct, blockify, deblockify

from .fuse import aot_fuse

aot_block_dct = aot_fuse(block_dct)
aot_block_idct = aot_fuse(block_idct)


class BlockDCTSandwich(nn.Module):
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
            idx, inv_idx = self.build_idx()
            self.idx = nn.Parameter(idx, requires_grad=False)
            self.inv_idx = nn.Parameter(inv_idx, requires_grad=False)

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
        return out, inv_out

    def to_zigzag(self, x):
        bsz, ch, n = x.shape[:3]
        x = x.view(bsz, ch, n, self.block_size**2)
        idx = self.idx.expand_as(x)
        return x.gather(-1, idx).contiguous()

    def from_zigzag(self, x):
        bsz, ch, n = x.shape[:3]
        x = x.view(bsz, ch, n, self.block_size**2)
        inv_idx = self.inv_idx.expand_as(x)
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


class FreqConv(nn.Module):
    def __init__(self, in_ch, out_ch, window_size, groups=1):
        super().__init__()
        self.padding = window_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, window_size, padding=0, groups=groups)

    def forward(self, x):
        bsz, ch, n, dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz * n, ch, dim)
        x = F.pad(x, (0, self.padding))
        x = self.conv(x)
        x = x.view(bsz, n, -1, x.shape[-1])
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class DCTFreqConv(nn.Module):
    def __init__(self, in_ch, out_ch, window_size: int, block_size: int, groups=1):
        super().__init__()
        conv = FreqConv(in_ch, out_ch, window_size=window_size, groups=groups)
        self.sand = BlockDCTSandwich(conv, block_size, zigzag=True)

    def forward(self, x):
        return self.sand(x)
