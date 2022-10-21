import torch
import torch.nn.functional as F
from torch import nn
from torchjpeg.dct import block_dct, block_idct, blockify, deblockify

from .ckpt import ckpt_forward


class BlockDCTSandwich(nn.Module):
    def __init__(self, module: nn.Module, block_size: int):
        """DCT -> module -> IDCT

        Args:
            module (nn.Module): Module to apply to DCT coefficients with the shape of (bsz, ch, n_blocks, block_size ** 2)
            block_size (int): Size of the block to use for DCT
        """
        super().__init__()
        self.module = module
        self.block_size = block_size
        self.idx = nn.Parameter(self.build_idx(), requires_grad=False)

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
        return out.view(1, 1, 1, -1)

    @ckpt_forward
    def forward(self, x):
        # nchw->nclb^2 where b = block_size
        bsz, ch, *size = x.shape
        x = blockify(x, self.block_size)
        x = block_dct(x)
        x = x.reshape(bsz, ch, -1, self.block_size**2)
        idx = self.idx.expand_as(x)
        x = x.gather(-1, idx).contiguous()
        x = self.module(x)
        # nclb^2->nchw
        assert x.shape[-1] == self.block_size**2
        ch = x.shape[1]
        x = x.view(bsz, ch, -1, self.block_size, self.block_size)
        x = block_idct(x)
        x = deblockify(x, size)
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
        self.sand = BlockDCTSandwich(conv, block_size)

    def forward(self, x):
        return self.sand(x)
