import torch
import torch.nn.functional as F


def soft_clip(x, lb, ub):
    y = torch.clamp(x, lb, ub)
    if x.requires_grad:
        y = y.detach() + x - x.detach()
    return y


def RGB2YCbCr(rgb):
    # (batch, channel, height, width)
    A = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    bias = torch.tensor([0.0, 0.5, 0.5], dtype=rgb.dtype, device=rgb.device)
    scale = torch.tensor([1.0, 0.872, 1.23], dtype=rgb.dtype, device=rgb.device)
    yuv = torch.einsum("bchw,cd->bdhw", rgb, A)
    ycbcr = yuv * scale.view(1, 3, 1, 1) + bias.view(1, 3, 1, 1)
    ycbcr = soft_clip(ycbcr, 0.0, 1.0)
    return ycbcr


def YCbCr2RGB(ycbcr):
    # (batch, channel, height, width)
    bias = torch.tensor([0.0, 0.5, 0.5], dtype=ycbcr.dtype, device=ycbcr.device)
    scale = torch.tensor([1.0, 0.872, 1.23], dtype=ycbcr.dtype, device=ycbcr.device)

    A = torch.tensor(
        [
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0],
        ],
        dtype=ycbcr.dtype,
        device=ycbcr.device,
    )
    yuv = (ycbcr - bias.view(1, 3, 1, 1)) / scale.view(1, 3, 1, 1)
    rgb = torch.einsum("bchw,cd->bdhw", yuv, A)
    rgb = soft_clip(rgb, 0.0, 1.0)
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
    cb = F.interpolate(cb, scale_factor=2, mode="nearest")
    cr = F.interpolate(cr, scale_factor=2, mode="nearest")
    cbcr = torch.cat([cb, cr], dim=1)
    ycbcr = torch.cat([l, cbcr], dim=1)
    rgb = YCbCr2RGB(ycbcr)
    return rgb
