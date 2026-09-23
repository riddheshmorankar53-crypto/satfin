"""Backward warping with grid_sample."""
import torch
import torch.nn.functional as F


def backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample img (B,C,H,W) at p + flow(p); flow (B,2,H,W) is in pixels, channel 0 = x, 1 = y."""
    B, _, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    x = xs + flow[:, 0]
    y = ys + flow[:, 1]
    grid = torch.stack((2 * x / max(W - 1, 1) - 1, 2 * y / max(H - 1, 1) - 1), dim=-1)  # (B,H,W,2)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
