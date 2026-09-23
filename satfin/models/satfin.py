"""SatFIN = IFNet flow + mask blend + fusion U-Net residual.

python -m satfin.models.satfin      # parameter count + dummy forward pass
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from satfin.models.fusion import FusionUNet
from satfin.models.ifnet import IFNet
from satfin.models.warp import backward_warp


class SatFIN(nn.Module):
    """Interpolate frame at time t in (0,1) between I0 and I1 (B,C,H,W in [0,1]); any H, W."""

    def __init__(self, channels: int = 1, ifnet_widths: tuple[int, ...] = (96, 64, 48),
                 scales: tuple[int, ...] = (4, 2, 1), fusion_width: int = 32) -> None:
        super().__init__()
        self.ifnet = IFNet(channels, tuple(ifnet_widths), tuple(scales))
        self.fusion = FusionUNet(channels, fusion_width)
        self.multiple = 4 * max(scales)

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns pred (final frame), merged (blend before refinement), flow (B,4,H,W), mask (B,1,H,W)."""
        H, W = I0.shape[-2:]
        m = self.multiple
        pad = (0, (-W) % m, 0, (-H) % m)
        # BT frames are low-contrast (std ~0.03); standardize per sample so the net sees O(1) signals
        both = torch.cat([I0, I1], 1).flatten(1)
        mu = both.mean(1).view(-1, 1, 1, 1)
        sd = both.std(1).clamp_min(1e-3).view(-1, 1, 1, 1)
        I0p = F.pad((I0 - mu) / sd, pad, mode="replicate")
        I1p = F.pad((I1 - mu) / sd, pad, mode="replicate")
        flow, mask_logit = self.ifnet(I0p, I1p, t)
        mask = torch.sigmoid(mask_logit)
        w0, w1 = backward_warp(I0p, flow[:, :2]), backward_warp(I1p, flow[:, 2:])
        merged = mask * w0 + (1 - mask) * w1
        pred = merged + self.fusion(I0p, I1p, w0, w1, merged, mask, flow)
        back = lambda x: (x[..., :H, :W] * sd + mu).clamp(0, 1)
        crop = lambda x: x[..., :H, :W]
        return {"pred": back(pred), "merged": back(merged), "flow": crop(flow), "mask": crop(mask)}


def build_model(cfg: dict) -> SatFIN:
    """Build SatFIN from the `model` section of the config."""
    return SatFIN(cfg["channels"], cfg["ifnet_widths"], cfg["scales"], cfg["fusion_width"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from satfin.config import load_config
    from satfin.env_check import get_device

    dev = get_device()
    cfg = load_config()["model"]
    model = build_model(cfg).to(dev).eval()
    print(f"SatFIN: {count_params(model) / 1e6:.2f} M params "
          f"(IFNet {count_params(model.ifnet) / 1e6:.2f} M, fusion {count_params(model.fusion) / 1e6:.2f} M)")
    c = cfg["channels"]
    for size in (256, 500):
        x0, x1 = torch.rand(1, c, size, size, device=dev), torch.rand(1, c, size, size, device=dev)
        with torch.no_grad():
            t0 = time.time()
            out = model(x0, x1, torch.tensor([[0.5]], device=dev))
        print(f"{size}x{size} on {dev}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in out.items())
              + f"  {time.time() - t0:.2f}s")
