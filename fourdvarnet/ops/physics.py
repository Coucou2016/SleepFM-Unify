"""Fixed finite-difference operators (non-trainable conv layers)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _central_kernel(axis: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if axis == "x":
        k = torch.tensor([[[[-1.0, 0.0, 1.0]]]], device=device, dtype=dtype) * 0.5
    else:
        k = torch.tensor([[[[-1.0], [0.0], [1.0]]]], device=device, dtype=dtype) * 0.5
    return k


def gradient_xy(field: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered differences on (B, C, H, W)."""
    b, c, _, _ = field.shape
    kx = _central_kernel("x", field.device, field.dtype).repeat(c, 1, 1, 1)
    ky = _central_kernel("y", field.device, field.dtype).repeat(c, 1, 1, 1)
    gx = F.conv2d(field, kx, padding=(0, 1), groups=c) / dx
    gy = F.conv2d(field, ky, padding=(1, 0), groups=c) / dy
    return gx, gy


def divergence(u: torch.Tensor, v: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    du_dx, _ = gradient_xy(u, dx=dx, dy=dy)
    _, dv_dy = gradient_xy(v, dx=dx, dy=dy)
    return du_dx + dv_dy


def geostrophic_uv_from_ssh(
    ssh: torch.Tensor,
    f_coriolis: float,
    g: float = 9.81,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper Eq. 5: u_g = -(g/f) d_y SSH, v_g = (g/f) d_x SSH."""
    dssh_dx, dssh_dy = gradient_xy(ssh, dx=dx, dy=dy)
    u_g = -(g / f_coriolis) * dssh_dy
    v_g = (g / f_coriolis) * dssh_dx
    return u_g, v_g
