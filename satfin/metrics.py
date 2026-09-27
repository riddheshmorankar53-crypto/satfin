"""Image-quality metrics on normalized [0,1] frames: torch PSNR for training, numpy scores for evaluation."""
import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-sample PSNR (dB) for (B,...) tensors with data range 1."""
    mse = ((pred - gt) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return -10 * torch.log10(mse)


def scores(pred: np.ndarray, gt: np.ndarray, span: float, cold: float) -> tuple[float, float, float, float]:
    """PSNR (dB), SSIM, MAE (K), and MAE (K) on convective pixels (gt < cold, normalized; NaN if none).

    `span` = bt_max - bt_min converts normalized errors to kelvin.
    """
    err = np.abs(pred - gt)
    mse = max(float((err ** 2).mean()), 1e-12)
    c = gt < cold
    return (-10 * np.log10(mse), structural_similarity(pred, gt, data_range=1.0),
            float(err.mean()) * span, float(err[c].mean()) * span if c.any() else float("nan"))
