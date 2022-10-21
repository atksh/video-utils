import math

import torch
import torch.nn.functional as F
import torchjpeg
from torch import nn


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
        self.idx = self.build_idx()

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
        out = torch.tensor(out)
        return out

    def forward(self, x):
        # nchw->nclb^2 where b = block_size
        idx = self.idx.to(x.device)
        bsz, ch, *size = x.shape
        x = torchjpeg.dct.blockify(x, self.block_size)
        x = torchjpeg.dct.block_dct(x)
        x = x[:, :, idx[:, 0], idx[:, 1]]  # sig-zag indexing
        x = x.reshape(bsz, ch, -1, self.block_size**2)
        x = self.module(x)
        # nclb^2->nchw
        ch = x.shape[1]
        x = x.view(bsz, ch, -1, self.block_size, self.block_size)
        x = torchjpeg.dct.block_idct(x)
        x = torchjpeg.dct.deblockify(x, size)
        return x


class FreqConv(nn.Module):
    def __init__(self, in_ch, out_ch, window_size):
        super().__init__()
        self.padding = math.ceil(window_size / 2)
        self.conv = nn.Conv1d(in_ch, out_ch, window_size, padding=0)

    def forward(self, x):
        bsz, ch, n, dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz * n, ch, dim)
        x = F.pad(x, (0, self.padding))
        x = self.conv(x)
        x = x.view(bsz, n, -1, x.shape[-1])
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class DCTFreqConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, block_size: int):
        super().__init__()
        self.conv = FreqConv(in_ch=in_ch, out_ch=out_ch, window_size=window_size)
        self.sand = BlockDCTSandwich(self.w, block_size)

    def forward(self, x):
        return self.sand(x)
