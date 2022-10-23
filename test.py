import sys

import torch

sys.path.append("./src")
from video.nn.module import Decoder, Encoder

in_ch = out_ch = 3
widths = [8, 16, 32, 64]
depths = [2, 2, 4, 2]
heads = [1, 2, 4, 8]
head_widths = [8, 8, 8, 8]
block_sizes = [8, 8, 8, 8]
kernel_sizes = [3, 3, 3, 3]
dec_depths = [1, 1, 1]
in_widths = widths[1:][::-1]
add_widths = widths[:-1][::-1]
out_widths = widths[:-1][::-1]
dec_heads = heads[:-1][::-1]
dec_head_widths = head_widths[:-1][::-1]
dec_block_sizes = block_sizes[:-1][::-1]
dec_kernel_sizes = kernel_sizes[:-1][::-1]


enc = Encoder(
    in_ch=in_ch,
    widths=widths,
    depths=depths,
    heads=heads,
    head_widths=head_widths,
    block_sizes=block_sizes,
    kernel_sizes=kernel_sizes,
)

dec = Decoder(
    out_ch=out_ch,
    in_widths=in_widths,
    add_widths=add_widths,
    out_widths=out_widths,
    depths=dec_depths,
    heads=dec_heads,
    head_widths=dec_head_widths,
    block_sizes=dec_block_sizes,
    kernel_sizes=dec_kernel_sizes,
)

x = torch.randn(1, 7, 3, 448, 252)
feats = enc(x)
y = dec(x, feats)

print(x.shape, y.shape)
print(y)
