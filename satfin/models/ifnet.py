"""IFNet: coarse-to-fine bidirectional intermediate-flow estimator (RIFE-style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from satfin.models.warp import backward_warp


def conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.PReLU(cout))


class IFBlock(nn.Module):
    """One scale: sees inputs downsampled by `scale`, predicts a flow update (4ch) and mask update (1ch)."""

    def __init__(self, cin: int, width: int, scale: int, depth: int = 4) -> None:
        super().__init__()
        self.scale = scale
        self.down = nn.Sequential(conv(cin, width // 2, 2), conv(width // 2, width, 2))
        self.body = nn.Sequential(*[conv(width, width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 5, 3, 1, 1)

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = x.shape[-2:]
        s = self.scale
        if flow is not None:
            x = torch.cat([x, flow / s], 1)  # flow in pixels of the downsampled grid
        if s != 1:
            x = F.interpolate(x, scale_factor=1 / s, mode="bilinear", align_corners=False)
        f = self.down(x)
        f = self.body(f) + f
        out = F.interpolate(self.head(f), size=(H, W), mode="bilinear", align_corners=False)
        return out[:, :4] * s * 4, out[:, 4:5]  # head works at 1/(4s) resolution -> pixels at full res


class IFNet(nn.Module):
    """Takes I0, I1 (B,C,H,W) and t (B,1); returns flow (B,4,H,W) = [F_t->0, F_t->1] and mask logits (B,1,H,W).

    H and W must be divisible by 4 * max(scales).
    """

    def __init__(self, channels: int = 1, widths: tuple[int, ...] = (96, 64, 48),
                 scales: tuple[int, ...] = (4, 2, 1)) -> None:
        super().__init__()
        first_in = 2 * channels + 1                     # I0, I1, t
        next_in = 4 * channels + 1 + 1 + 4              # I0, I1, warped I0, warped I1, t, mask, flow
        self.blocks = nn.ModuleList(
            IFBlock(first_in if i == 0 else next_in, w, s) for i, (w, s) in enumerate(zip(widths, scales))
        )

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tmap = t.view(-1, 1, 1, 1).expand(-1, 1, *I0.shape[-2:])
        flow, mask = self.blocks[0](torch.cat([I0, I1, tmap], 1), None)
        for block in self.blocks[1:]:
            w0 = backward_warp(I0, flow[:, :2])
            w1 = backward_warp(I1, flow[:, 2:])
            df, dm = block(torch.cat([I0, I1, w0, w1, tmap, mask], 1), flow)
            flow, mask = flow + df, mask + dm
        return flow, mask
