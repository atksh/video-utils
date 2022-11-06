import enum
from typing import List, Optional, Tuple, Union

import revlib
import torch
import torch.nn.functional as F
from functorch.compile import memory_efficient_fusion
from torch import nn
from torch.jit import Final
from torchtyping import TensorType as TT

ChannelTensor = TT["batch", "channel", "length"]
ImageTensor = TT["batch", "channel", "height", "width"]
VideoTensor = TT["batch", "time", "channel", "height", "width"]


class NonLinear(nn.Module):
    def forward(self, x: TT[...]) -> TT[...]:
        return F.leaky_relu(x, 0.01)


class SELayer(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            NonLinear(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: ImageTensor) -> ImageTensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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
    dilation: Final[int]

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(dim, dim, 2, bias=False, padding=0, dilation=dilation)
        self.pad = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: ChannelTensor) -> ChannelTensor:
        b = x.shape[0]
        pad = self.pad.expand(b, -1, self.dilation)
        x = torch.cat([pad, x], dim=2)
        x = self.conv(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, mul: int = 4):
        super().__init__()
        self.wi = nn.Conv1d(dim, dim * mul, 1)
        self.wo = nn.Conv1d(dim * mul // 2, dim, 1)
        self.act = NonLinear()

    def forward(self, x: ChannelTensor) -> ChannelTensor:
        x1, x2 = self.wi(x).chunk(2, dim=1)
        x = x1 * self.act(x2)
        x = self.wo(x)
        return x


class LinearAttention(nn.Module):
    heads: Final[int]

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(
        self, x: Union[ChannelTensor, ImageTensor]
    ) -> Union[ChannelTensor, ImageTensor]:
        b, _, *size = x.shape
        if len(size) == 2:
            l = size[0] * size[1]
        elif len(size) == 1:
            l = size[0]
        x = x.view(b, -1, l)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.view(b, self.heads, -1, l)
        k = k.view(b, self.heads, -1, l).softmax(dim=-1)
        v = v.view(b, self.heads, -1, l)
        # bhdn,bhen->bhde
        context = torch.matmul(k, v.transpose(-1, -2))
        # bhde,bhdn->bhen
        out = torch.matmul(context.transpose(-1, -2), q)
        out = self.to_out(out.view(b, -1, l))
        out = out.view(b, -1, *size)
        return out


class LayerNorm(nn.Module):
    eps: Final[float]
    normalized_shape: Final[Tuple[int]]

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: ChannelTensor) -> ChannelTensor:
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        x = x.permute(0, 2, 1)
        return x


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 2, 2, bias=False)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: ImageTensor) -> ImageTensor:
        return self.norm(self.conv(x))


class Downsample3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.down = Downsample(dim)

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t = x.shape[:2]
        shape = x.shape[2:]
        x = x.view(b * t, *shape)
        x = self.down(x)
        shape = x.shape[1:]
        x = x.view(b, t, *shape)
        return x


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
        shape = x.shape[2:]
        x = x.view(b * t, *shape)
        shape = y.shape[2:]
        y = y.view(b * t, *shape)
        x = self.up(x, y)
        shape = x.shape[1:]
        x = x.view(b, t, *shape)
        return x


class LayerScale(nn.Module):
    def __init__(self, initial_scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

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
    block_size: Final[Tuple[int, int]]

    def __init__(self, block_size: Union[int, Tuple[int]]):
        super().__init__()
        if isinstance(block_size, int):
            block_size = (block_size, block_size)
        self.block_size = block_size

    def forward(
        self, x: ImageTensor
    ) -> TT["batch", "channel", "num_blocks", "block_heigh", "block_width"]:
        b, c, h, w = x.shape
        x = x.contiguous().view(b * c, 1, h, w)
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

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t, _, *size = x.shape
        shape = x.shape[2:]
        x = x.view(b * t, *shape)
        x = self.blockify(x)
        x = x.transpose(1, 2).contiguous()
        shape = x.shape[2:]
        x = x.view(-1, *shape)  # (b * n, c, h, w)
        x = self.module(x)
        shape = x.shape[1:]
        x = x.view(b * t, -1, *shape)  # (b, n, c, h, w)
        x = x.transpose(1, 2)  # (b, c, n, h, w)
        x = self.unblockify(x, size)
        shape = x.shape[1:]
        x = x.view(b, t, *shape)
        return x


class ImageWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t, _, h, w = x.shape
        x = x.view(b * t, -1, h, w)
        x = self.module(x)
        shape = x.shape[1:]
        x = x.view(b, t, *shape)
        return x


class HeightWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t, _, h, w = x.shape
        x = x.view(b * t, -1, h, w)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b * t * h, -1, w)
        x = self.module(x)
        shape = x.shape[1:]
        x = x.view(b * t, h, *shape)
        x = x.permute(0, 2, 1, 3)
        shape = x.shape[1:]
        x = x.view(b, t, *shape)
        return x


class WidthWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t, _, h, w = x.shape
        x = x.view(b * t, -1, h, w)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b * t * w, -1, h)
        x = self.module(x)
        shape = x.shape[1:]
        x = x.view(b * t, w, *shape)
        x = x.permute(0, 2, 3, 1)
        shape = x.shape[1:]
        x = x.view(b, t, *shape)
        return x


class ChannelWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, t, _, h, w = x.shape
        x = x.view(b * t, -1, h * w)
        x = self.module(x)
        x = x.view(b, t, -1, h, w)
        return x


class TimeWise(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: VideoTensor) -> VideoTensor:
        b, _, _, h, w = x.shape
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        shape = x.shape[3:]
        x = x.view(b * h * w, *shape)
        x = self.module(x)
        shape = x.shape[1:]
        x = x.view(b, h, w, *shape)
        x = x.permute(0, 4, 3, 1, 2)
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
        return BlockWise(block_size, module)
    elif layer_type == LayerType.height:
        return HeightWise(module)
    elif layer_type == LayerType.width:
        return WidthWise(module)
    elif layer_type == LayerType.channel:
        return ChannelWise(module)
    else:
        raise ValueError(f"Unknown layer type {layer_type}")


def make_block(
    module: nn.Module,
    dim: int,
    layer_type: LayerType,
    block_size: int = 8,
    eps: float = 1e-5,
    initial_scale: float = 0.1,
    drop_prob: float = 0.0,
    add_prenorm: bool = True,
    add_layer_scale: bool = True,
    add_droppath: bool = True,
) -> nn.Module:
    out = []
    if add_prenorm:
        out.append(wrap_layer(LayerNorm(dim, eps), LayerType.channel, block_size))
    out.append(wrap_layer(module, layer_type, block_size))
    if add_layer_scale:
        out.append(LayerScale(initial_scale))
    if add_droppath and drop_prob > 0.0:
        out.append(DropPath(drop_prob))
    if len(out) == 1:
        return out[0]
    else:
        return nn.Sequential(*out)


class RevResidualSequential(nn.Module):
    split_dim: Final[int]

    def __init__(self, layers, split_dim, fuse: bool = True):
        super().__init__()
        self.split_dim = split_dim
        if fuse:
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


class NaiveResidualSequential(nn.Module):
    def __init__(self, layers, fuse: bool = True):
        super().__init__()
        if fuse:
            layers = [memory_efficient_fusion(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: TT[...]) -> TT[...]:
        for layer in self.layers:
            x = x + layer(x)
        return x


class ResidualSequential(nn.Module):
    def __init__(
        self,
        layers,
        split_dim,
        fuse: bool = False,
        reversible: bool = False,
    ):
        super().__init__()
        if reversible:
            self.module = RevResidualSequential(layers, split_dim, fuse=fuse)
        else:
            self.module = NaiveResidualSequential(layers, fuse=fuse)

    def forward(self, x: TT[...]) -> TT[...]:
        return self.module(x)


class Stage(nn.Module):
    dim: Final[int]
    depth: Final[int]
    block_size = Final[int]
    eps: Final[float]
    initial_scale: Final[float]
    drop_prob: Final[float]

    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int = 7,
        heads: int = 4,
        head_dim: int = 32,
        block_size: int = 8,
        eps: float = 1e-5,
        initial_scale: float = 0.1,
        drop_prob: float = 0.0,
        fuse: bool = False,
        reversible: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.block_size = block_size
        self.eps = eps
        self.initial_scale = initial_scale
        self.drop_prob = drop_prob

        layers = []
        for _ in range(depth):
            for dilation in [1, 2, 4]:
                layers.append(
                    (
                        LayerType.time,
                        TimeConv(dim, dilation=dilation),
                        False,
                        False,
                        False,
                    )
                )
            layers.append(
                (
                    LayerType.image,
                    SameConv2d(dim, dim, kernel_size, dim),
                    False,
                    False,
                    False,
                )
            )
            # layers.append(
            #     (
            #         LayerType.time,
            #         LinearAttention(dim, heads, head_dim),
            #         True,
            #         True,
            #         True,
            #     )
            # )
            # layers.append(
            #     (
            #         LayerType.block,
            #         LinearAttention(dim, heads, head_dim),
            #         True,
            #         True,
            #         True,
            #     )
            # )
            # layers.append(
            #     (
            #         LayerType.height,
            #         LinearAttention(dim, heads, head_dim),
            #         True,
            #         True,
            #         True,
            #     )
            # )
            # layers.append(
            #     (
            #         LayerType.width,
            #         LinearAttention(dim, heads, head_dim),
            #         True,
            #         True,
            #         True,
            #     )
            # )
            layers.append((LayerType.channel, FeedForward(dim), True, True, True))
            layers.append(
                (
                    LayerType.image,
                    SELayer(dim),
                    False,
                    False,
                    False,
                )
            )
            # layers.append(
            #     (
            #         LayerType.block,
            #         SELayer(dim),
            #         False,
            #         False,
            #         False,
            #     )
            # )

        layers = [self.make_block(*args) for args in layers]
        self.layers = ResidualSequential(
            layers, split_dim=2, fuse=fuse, reversible=reversible
        )
        self.norm = ChannelWise(LayerNorm(dim, eps))

    def make_block(
        self, layer_type, module, add_prenorm, add_layer_scale, add_droppath
    ):
        return make_block(
            module,
            self.dim,
            layer_type,
            self.block_size,
            self.eps,
            self.initial_scale,
            self.drop_prob,
            add_prenorm,
            add_layer_scale,
            add_droppath,
        )

    def forward(self, x: VideoTensor) -> VideoTensor:
        x = self.layers(x)
        x = self.norm(x)
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
        eps: float = 1e-5,
        initial_scale: float = 0.1,
        drop_prob: float = 0.0,
        fuse: bool = False,
        reversible: bool = False,
    ):
        super().__init__()
        self.proj = ImageWise(SameConv2d(in_dim, out_dim, 1, bias=True))
        self.down = Downsample3D(out_dim)
        self.stage = Stage(
            out_dim,
            depth,
            kernel_size,
            heads,
            head_dim,
            block_size,
            eps,
            initial_scale,
            drop_prob,
            fuse,
            reversible,
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
        eps: float = 1e-5,
        initial_scale: float = 0.1,
        drop_prob: float = 0.0,
        fuse: bool = False,
        reversible: bool = False,
    ):
        super().__init__()
        self.up = Upsample3D(in_dim)
        self.proj = ImageWise(SameConv2d(in_dim + add_dim, out_dim, 1, bias=True))
        self.stage = Stage(
            out_dim,
            depth,
            kernel_size,
            heads,
            head_dim,
            block_size,
            eps,
            initial_scale,
            drop_prob,
            fuse,
            reversible,
        )

    def forward(self, x: VideoTensor, y: VideoTensor) -> VideoTensor:
        x = self.up(x, y)
        x = self.proj(x)
        x = self.stage(x)
        return x


class Encoder(nn.Module):
    resolution_scale: Final[int]

    def __init__(
        self,
        in_ch: int,
        widths: List[int],
        depths: List[int],
        heads: List[int],
        head_widths: List[int],
        block_sizes: List[int],
        kernel_sizes: List[int],
        eps: float = 1e-5,
        initial_scale: float = 0.1,
        drop_prob: float = 0.0,
        resolution_scale: int = 1,
        fuse: bool = False,
        reversible: bool = False,
    ):
        super().__init__()
        self.resolution_scale = resolution_scale
        in_widths = [in_ch * (self.resolution_scale**2)] + widths[:-1]
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
                    initial_scale,
                    drop_prob,
                    fuse,
                    reversible,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.resize = ImageWise(nn.PixelUnshuffle(self.resolution_scale))

    def forward(self, x: VideoTensor) -> List[VideoTensor]:
        x = self.resize(x)
        feats = [x]
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class Decoder(nn.Module):
    resolution_scale: Final[int]

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
        eps: float = 1e-5,
        initial_scale: float = 0.1,
        drop_prob: float = 0.0,
        resolution_scale: int = 1,
        fuse: bool = False,
        reversible: bool = False,
    ):
        super().__init__()
        self.resolution_scale = resolution_scale
        add_widths[-1] = add_widths[-1] * (self.resolution_scale**2)
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
                    initial_scale,
                    drop_prob,
                    fuse,
                    reversible,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.resize = ImageWise(
            nn.Sequential(
                SameConv2d(
                    out_widths[-1],
                    out_ch * (self.resolution_scale**2),
                    out_ch,
                    1,
                    bias=True,
                ),
                nn.PixelShuffle(self.resolution_scale),
            )
        )

    def scale(self, x: VideoTensor, size: Tuple[int, int]) -> VideoTensor:
        b, t = x.shape[:2]
        shape = x.shape[2:]
        x = x.view(b * t, *shape)
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)
        shape = x.shape[1:]
        x = x.view(b, t, *shape)
        return x

    def duplicate_last(self, x: VideoTensor) -> VideoTensor:
        return torch.cat([x, x[:, [-1]]], dim=1)

    def soft_clip(self, x: VideoTensor) -> VideoTensor:
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        _x = x.clip(-1, 1)
        x = _x.detach() + x - x.detach()
        x = 0.5 * x + 0.5
        return x

    def forward(self, feats: List[VideoTensor], size: Tuple[int, int]) -> VideoTensor:
        feats = [self.duplicate_last(feat) for feat in feats]
        z = feats[-1]
        feats = feats[::-1][1:]
        for stage, feat in zip(self.stages, feats):
            z = stage(z, feat)

        z = z[:, [-1]]
        z = self.resize(z)
        z = self.scale(z, size)
        z = self.soft_clip(z)
        return z


class VideoModel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        widths: List[int],
        depths: List[int],
        heads: List[int],
        head_widths: List[int],
        block_sizes: List[int],
        kernel_sizes: List[int],
        dec_depths: List[int],
        resolution_scale: int = 1,
        fuse: bool = False,
        reversible: bool = False,
    ):
        super().__init__()
        in_widths = widths[::-1]
        add_widths = widths[:-1][::-1] + [in_ch]
        out_widths = in_widths[1:] + [in_widths[-1]]
        dec_heads = heads[:-1][::-1] + [heads[0]]
        dec_head_widths = head_widths[:-1][::-1] + [head_widths[0]]
        dec_block_sizes = block_sizes[:-1][::-1] + [block_sizes[0]]
        dec_kernel_sizes = kernel_sizes[:-1][::-1] + [kernel_sizes[0]]

        self.encoder = Encoder(
            in_ch=in_ch,
            widths=widths,
            depths=depths,
            heads=heads,
            head_widths=head_widths,
            block_sizes=block_sizes,
            kernel_sizes=kernel_sizes,
            resolution_scale=resolution_scale,
            fuse=fuse,
            reversible=reversible,
        )

        self.decoder = Decoder(
            out_ch=out_ch,
            in_widths=in_widths,
            add_widths=add_widths,
            out_widths=out_widths,
            depths=dec_depths,
            heads=dec_heads,
            head_widths=dec_head_widths,
            block_sizes=dec_block_sizes,
            kernel_sizes=dec_kernel_sizes,
            resolution_scale=resolution_scale,
            fuse=fuse,
            reversible=reversible,
        )

    def forward(self, x: VideoTensor) -> VideoTensor:
        size = x.shape[-2:]
        feats = self.encoder(x)
        return self.decoder(feats, size)


class ModelWithLoss(nn.Module):
    def __init__(self, model: nn.Module, loss: nn.Module):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, x: VideoTensor, y: Optional[VideoTensor] = None) -> VideoTensor:
        y_hat = self.model(x)
        if y is not None:
            return self.loss(y_hat, y)
        else:
            return y_hat


def make_fused_model_loss(model: nn.Module, loss: nn.Module):
    # return memory_efficient_fusion(ModelWithLoss(model, loss))
    return ModelWithLoss(model, loss)
