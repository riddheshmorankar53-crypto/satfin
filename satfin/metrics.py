"""Image-quality metrics on normalized [0,1] frames."""
import torch


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-sample PSNR (dB) for (B,...) tensors with data range 1."""
    mse = ((pred - gt) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return -10 * torch.log10(mse)
