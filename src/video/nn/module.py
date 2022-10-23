import enum
import math
from typing import List, Tuple, Union

import revlib
import torch
import torch.nn.functional as F
from functorch.compile import memory_efficient_fusion
from torch import nn
from torchtyping import TensorType as TT

ChannelTensor = TT["batch", "channel", "length"]
ImageTensor = TT["batch", "channel", "height", "width"]
VideoTensor = TT["batch", "time", "channel", "height", "width"]


class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: ImageTensor) -> ImageTensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SameConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch,
        kernel_size: int = 3,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.pad = nn.ReflectionPad2d((kernel_size - 1) // 2)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=bias
        )

    def forward(self, x: ImageTensor) -> ImageTensor:
        return self.conv(self.pad(x))


class TimeConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 2, bias=False, padding=0)

    def forward(self, x: ChannelTensor) -> ChannelTensor:
        x = F.pad(x, (1, 0))
        return self.conv(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mul: int = 4):
        super().__init__()
        self.wi = nn.Conv1d(dim, dim * mul, 1)
        self.wo = nn.Conv1d(dim * mul // 2, dim, 1)

    def forward(self, x: ChannelTensor) -> ChannelTensor:
        x1, x2 = self.wi(x).chunk(2, dim=1)
        x = x1 * F.silu(x2)
        x = self.wo(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: ChannelTensor) -> ChannelTensor:
        b, _, l = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.view(b, self.heads, -1, l)
        k = k.view(b, self.heads, -1, l).softmax(dim=-1)
        v = v.view(b, self.heads, -1, l)
        # bhdn,bhen->bhde
        context = torch.matmul(k, v.transpose(-1, -2))
        # bhde,bhdn->bhen
        out = torch.matmul(context.transpose(-1, -2), q)
        out = self.to_out(out.view(b, -1, l))
        return out


class Norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = math.sqrt(dim)
        self.gamma = nn.Parameter(torch.ones((1, dim, 1)))
        self.beta = nn.Parameter(torch.zeros((1, dim, 1)))

    def forward(self, x: ChannelTensor) -> ChannelTensor:
        x = F.normalize(x, dim=1, eps=self.eps)
        x = x * self.gamma * self.scale + self.beta
        return x


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 2, 2, bias=False)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: ImageTensor) -> ImageTensor:
        return self.norm(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 2, 2, bias=False)

    def forward(self, x: ImageTensor, y: ImageTensor) -> ImageTensor:
        size = y.shape[-2:]
        x = self.conv(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return torch.cat([x, y], dim=1)


class Upsample3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.up = Upsample(dim)

    def forward(self, x: VideoTensor, y: VideoTensor) -> VideoTensor:
        b, t = x.shape[:2]
        x = x.view(b * t, *x.shape[2:])
        y = y.view(b * t, *y.shape[2:])
        x = self.up(x, y)
        x = x.view(b, t, *x.shape[1:])
        return x


class LayerScale(nn.Module):
    def __init__(self, initial_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_value))

    def forward(self, x: TT[...]) -> TT[...]:
        return x * self.scale


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: TT[...]) -> TT[...]:
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            mask = torch.empty(x.shape[0], device=x.device).bernoulli_(keep_prob)
            mask = mask.view(-1, *([1] * (x.ndim - 1)))
            x = x / keep_prob * mask
        return x


class Blockify(nn.Module):
    def __init__(self, block_size: Union[int, Tuple[int]]):
        super().__init__()
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        self.block_size = block_size

    def forward(
        self, x: ImageTensor
    ) -> TT["batch", "channel", "num_blocks", "block_heigh", "block_width"]:
        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)
        x = F.unfold(x, self.block_size, stride=self.block_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, -1, *self.block_size)
        return x


class UnBlockify(nn.Module):
    def forward(
        self,
        x: TT["batch", "channel", "num_blocks", "block_heigh", "block_width"],
        size: Tuple[int, int],
    ) -> ImageTensor:
        b, c, n, h, w = x.shape
        block_size = (h, w)
        x = x.contiguous().view(b * c, n, h * w)
        x = x.transpose(1, 2)
        x = F.fold(x, size, block_size, stride=block_size)
        x = x.contiguous().view(b, c, *size)
        return x


class BlockWise(nn.Module):
    def __init__(self, block_size: Union[int, Tuple[int]], module: nn.Module):
        super().__init__()
        self.blockify = Blockify(block_size)
        self.module = module
        self.unblockify = UnBlockify()

    def forward(self, x: ImageTensor) -> ImageTensor:
        b, _, *size = x.shape
        x = self.blockify(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, *x.shape[2:])  # (b * n, c, h, w)
        x = self.module(x)
        x = x.view(b, -1, *x.shape[1:])  # (b, n, c, h, w)
        x = x.transpose(1, 2)  # (b, c, n, h, w)
        x = self.unblockify(x, size)
        return x


class ImageWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t, _, h, w = x.shape
        x = x.view(b * t, -1, h, w)
        x = self.module(x)
        x = x.view(b, t, *x.shape[1:])
        return x


class HeightWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: ImageTensor) -> ImageTensor:
        b, _, h, w = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b * h, -1, w)
        x = self.module(x)
        x = x.view(b, h, -1, w)
        x = x.permute(0, 2, 1, 3)
        return x


class WidthWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: ImageTensor) -> ImageTensor:
        b, _, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b * w, -1, h)
        x = self.module(x)
        x = x.view(b, w, -1, h)
        x = x.permute(0, 2, 3, 1)
        return x


class ChannelWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: ImageTensor) -> ImageTensor:
        b, _, h, w = x.shape
        x = x.view(b, -1, h * w)
        x = self.module(x)
        x = x.view(b, -1, h, w)
        return x


class TimeWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t, _, h, w = x.shape
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b * h * w, -1, t)
        x = self.module(x)
        x = x.view(b, h, w, -1, t).permute(0, 4, 3, 1, 2)
        return x


class LayerType(enum.Enum):
    time = enum.auto()
    image = enum.auto()
    block = enum.auto()
    height = enum.auto()
    width = enum.auto()
    channel = enum.auto()


def wrap_layer(
    module: nn.Module, layer_type: LayerType, block_size: int = 8
) -> nn.Module:
    if layer_type == LayerType.time:
        return TimeWise(module)
    elif layer_type == LayerType.image:
        return ImageWise(module)
    elif layer_type == LayerType.block:
        return ImageWise(BlockWise(block_size, module))
    elif layer_type == LayerType.height:
        return ImageWise(HeightWise(module))
    elif layer_type == LayerType.width:
        return ImageWise(WidthWise(module))
    elif layer_type == LayerType.channel:
        return ImageWise(ChannelWise(module))
    else:
        raise ValueError(f"Unknown layer type {layer_type}")


def make_block(
    module: nn.Module,
    dim: int,
    layer_type: LayerType,
    block_size: int = 8,
    eps: float = 1e-6,
    initial_value: float = 1.0,
    drop_prob: float = 0.0,
) -> nn.Module:
    return nn.Sequential(
        wrap_layer(Norm(dim, eps), LayerType.channel, block_size),
        wrap_layer(module, layer_type, block_size),
        LayerScale(initial_value),
        DropPath(drop_prob),
    )


class ResidualSequential(nn.Module):
    def __init__(self, layers, split_dim):
        super().__init__()
        self.split_dim = split_dim
        layers = [memory_efficient_fusion(layer) for layer in layers]
        self.layers = revlib.ReversibleSequential(*layers, split_dim=split_dim)
        self.s = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TT[...]) -> TT[...]:
        x = torch.cat([x, x], dim=self.split_dim)
        x1, x2 = self.layers(x).chunk(2, dim=self.split_dim)
        s = self.sigmoid(self.s)
        x = s * x1 + (1 - s) * x2
        return x


class Stage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int = 7,
        heads: int = 4,
        head_dim: int = 32,
        block_size: int = 8,
        eps: float = 1e-6,
        initial_value: float = 1.0,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.block_size = block_size
        self.eps = eps
        self.initial_value = initial_value
        self.drop_prob = drop_prob

        layers = []
        for _ in range(depth):
            layers.append((LayerType.time, TimeConv(dim))),
            layers.append((LayerType.image, SameConv2d(dim, dim, kernel_size, dim)))
            layers.append((LayerType.time, LinearAttention(dim, heads, head_dim)))
            layers.append(
                (LayerType.block, ChannelWise(LinearAttention(dim, heads, head_dim)))
            )
            layers.append((LayerType.height, LinearAttention(dim, heads, head_dim)))
            layers.append((LayerType.width, LinearAttention(dim, heads, head_dim)))
            layers.append((LayerType.channel, FeedForward(dim)))
            layers.append((LayerType.block, EfficientChannelAttention(kernel_size=7)))
            layers.append((LayerType.image, EfficientChannelAttention(kernel_size=3)))

        layers = [self.make_block(layer_type, module) for layer_type, module in layers]
        self.layers = ResidualSequential(layers, split_dim=2)

    def make_block(self, layer_type, module):
        return make_block(
            module,
            self.dim,
            layer_type,
            self.block_size,
            self.eps,
            self.initial_value,
            self.drop_prob,
        )

    def forward(self, x: VideoTensor) -> VideoTensor:
        x = self.layers(x)
        return x


class DownStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        kernel_size: int = 7,
        heads: int = 4,
        head_dim: int = 32,
        block_size: int = 8,
        eps: float = 1e-6,
        initial_value: float = 1.0,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.proj = ImageWise(SameConv2d(in_dim, out_dim, 1))
        self.down = ImageWise(Downsample(out_dim))
        self.stage = Stage(
            out_dim,
            depth,
            kernel_size,
            heads,
            head_dim,
            block_size,
            eps,
            initial_value,
            drop_prob,
        )

    def forward(self, x: VideoTensor) -> VideoTensor:
        x = self.proj(x)
        x = self.down(x)
        x = self.stage(x)
        return x


class UpStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        add_dim: int,
        out_dim: int,
        depth: int,
        kernel_size: int = 7,
        heads: int = 4,
        head_dim: int = 32,
        block_size: int = 8,
        eps: float = 1e-6,
        initial_value: float = 1.0,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.up = Upsample3D(in_dim)
        self.proj = ImageWise(SameConv2d(in_dim + add_dim, out_dim, 1))
        self.stage = Stage(
            out_dim,
            depth,
            kernel_size,
            heads,
            head_dim,
            block_size,
            eps,
            initial_value,
            drop_prob,
        )

    def forward(self, x: VideoTensor, y: VideoTensor) -> VideoTensor:
        x = self.up(x, y)
        x = self.proj(x)
        x = self.stage(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        widths: List[int],
        depths: List[int],
        heads: List[int],
        head_widths: List[int],
        block_sizes: List[int],
        kernel_sizes: List[int],
        eps: float = 1e-6,
        initial_value: float = 1.0,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        in_widths = [in_ch] + widths[:-1]
        stages = []
        for in_width, width, depth, head, head_width, block_size, kernel_size in zip(
            in_widths, widths, depths, heads, head_widths, block_sizes, kernel_sizes
        ):
            stages.append(
                DownStage(
                    in_width,
                    width,
                    depth,
                    kernel_size,
                    head,
                    head_width,
                    block_size,
                    eps,
                    initial_value,
                    drop_prob,
                )
            )

        self.stages = nn.ModuleList(stages)

    def forward(self, x: VideoTensor) -> List[VideoTensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class Decoder(nn.Module):
    def __init__(
        self,
        out_ch: int,
        in_widths: List[int],
        add_widths: List[int],
        out_widths: List[int],
        depths: List[int],
        heads: List[int],
        head_widths: List[int],
        block_sizes: List[int],
        kernel_sizes: List[int],
        eps: float = 1e-6,
        initial_value: float = 1.0,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        stages = []
        for (
            in_width,
            add_width,
            out_width,
            depth,
            head,
            head_width,
            block_size,
            kernel_size,
        ) in zip(
            in_widths,
            add_widths,
            out_widths,
            depths,
            heads,
            head_widths,
            block_sizes,
            kernel_sizes,
        ):
            stages.append(
                UpStage(
                    in_width,
                    add_width,
                    out_width,
                    depth,
                    kernel_size,
                    head,
                    head_width,
                    block_size,
                    eps,
                    initial_value,
                    drop_prob,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.fc = ImageWise(SameConv2d(out_widths[-1], out_ch, 1))

    def duplicate_last(self, x: VideoTensor) -> VideoTensor:
        return torch.cat([x, x[:, [-1]]], dim=1)

    def forward(self, x: VideoTensor, feats: List[VideoTensor]) -> VideoTensor:
        x = self.duplicate_last(x)
        feats = [self.duplicate_last(feat) for feat in feats]
        z = feats[-1]
        feats = feats[::-1][1:]
        for stage, feat in zip(self.stages, feats + [x]):
            z = stage(z, feat)
        z = z[:, [-1]]
        z = self.fc(z)
        return z.sigmoid()
