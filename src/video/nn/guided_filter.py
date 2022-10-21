import torch
from torch import nn
from torch.nn import functional as F


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        kernel_size = 2 * self.r + 1
        kernel_x = torch.full(
            (x.data.shape[1], 1, 1, kernel_size),
            1 / kernel_size,
            device=x.device,
            dtype=x.dtype,
        )
        kernel_y = torch.full(
            (x.data.shape[1], 1, kernel_size, 1),
            1 / kernel_size,
            device=x.device,
            dtype=x.dtype,
        )
        x = F.conv2d(x, kernel_x, padding=(0, self.r), groups=x.data.shape[1])
        x = F.conv2d(x, kernel_y, padding=(self.r, 0), groups=x.data.shape[1])
        return


class GuidedRefiner(nn.Module):
    def __init__(self, dim, z_dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.box_filter = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, bias=False, groups=dim
        )
        nn.init.constant_(self.box_filter.weight, 1 / 9)

        self.conv = nn.Sequential(
            nn.Conv2d(dim * 3 + z_dim, z_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, z_dim),
            nn.SiLU(),
            nn.Conv2d(z_dim, z_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, z_dim),
            nn.SiLU(),
            nn.Conv2d(z_dim, 3, kernel_size=1, bias=True),
        )

    def forward(self, lr_x, lr_y, hr_x, lr_z):
        assert lr_x.shape[1] == lr_y.shape[1] == hr_x.shape[1] == self.dim
        _, _, *size = hr_x.shape

        mean_x = self.box_filter(lr_x)
        mean_y = self.box_filter(lr_y)
        cov_xy = self.box_filter(lr_x * lr_y) - mean_x * mean_y
        var_x = self.box_filter(lr_x * lr_x) - mean_x * mean_x
        inv_var_x = torch.reciprocal(var_x + self.eps)
        A = cov_xy * inv_var_x
        A = self.conv(torch.cat([A, cov_xy, inv_var_x, lr_z], dim=1))
        b = mean_y - A * mean_x

        A = F.interpolate(A, size, mode="bilinear", align_corners=True)
        b = F.interpolate(b, size, mode="bilinear", align_corners=True)

        out = A * hr_x + b
        return out
