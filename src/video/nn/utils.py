import torch
import torch.nn.functional as F

from .layer import ImageToVideo, VideoToImage


@torch.jit.script
def gamma_correction(x):
    x1 = x / 12.92
    x2 = ((x + 0.055) / 1.055) ** 2.4
    x = torch.where(x <= 0.04045, x1, x2)
    return x


@torch.jit.script
def inverse_gamma_correction(x):
    x1 = x * 12.92
    x2 = 1.055 * x ** (1 / 2.4) - 0.055
    x = torch.where(x <= 0.0031308, x1, x2)
    return x


def soft_clip(x, lb, ub):
    y = torch.clamp(x, lb, ub)
    if x.requires_grad:
        y = y.detach() + x - x.detach()
    return y


def RGB2YCbCr(rgb):
    # (batch, channel, height, width)
    """Convert RGB to YCbCr. The input is expected to be in the range [0, 1]"""
    bias = torch.tensor([0.0, 0.5, 0.5]).view(1, 3, 1, 1)
    bias = bias.to(dtype=rgb.dtype, device=rgb.device)
    A = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    ycbcr = torch.einsum("bchw,cd->bdhw", rgb, A) + bias
    ycbcr = torch.clamp(ycbcr, 0.0, 1.0)
    return ycbcr


def YCbCr2RGB(ycbcr):
    # (batch, channel, height, width)
    """Convert YCbCr to RGB. The input is expected to be in the range [0, 1]"""
    bias = torch.tensor([0.0, 0.5, 0.5]).view(1, 3, 1, 1)
    bias = bias.to(dtype=ycbcr.dtype, device=ycbcr.device)
    A = torch.tensor(
        [
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0],
        ],
        dtype=ycbcr.dtype,
        device=ycbcr.device,
    )
    ycbcr = ycbcr - bias
    rgb = torch.einsum("bchw,cd->bdhw", ycbcr, A)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb


def to_YCbCr420(rgb):
    ycbcr = RGB2YCbCr(rgb)
    l = ycbcr[:, [0], :, :]
    cb = F.avg_pool2d(ycbcr[:, [1], :, :], kernel_size=2, stride=2)
    cr = F.avg_pool2d(ycbcr[:, [2], :, :], kernel_size=2, stride=2)
    cbcr = torch.cat([cb, cr], dim=1)
    return l, cbcr


def from_YCbCr420(l, cbcr):
    cb = cbcr[:, [0], :, :]
    cr = cbcr[:, [1], :, :]
    cb = F.interpolate(cb, scale_factor=2, mode="bilinear", align_corners=True)
    cr = F.interpolate(cr, scale_factor=2, mode="bilinear", align_corners=True)
    ycbcr = torch.cat([l, cb, cr], dim=1)
    rgb = YCbCr2RGB(ycbcr)
    return rgb


class RGB2YCbCr420(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    def forward(self, rgb):
        bsz = rgb.shape[0]
        rgb = self.to_image(rgb)
        l, cbcr = to_YCbCr420(rgb)
        l = self.to_video(l, bsz)
        cbcr = self.to_video(cbcr, bsz)
        return l, cbcr


class YCbCr420ToRGB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_image = VideoToImage()
        self.to_video = ImageToVideo()

    def forward(self, l, cbcr):
        bsz = l.shape[0]
        l = self.to_image(l)
        cbcr = self.to_image(cbcr)
        rgb = from_YCbCr420(l, cbcr)
        rgb = self.to_video(rgb, bsz)
        return rgb
