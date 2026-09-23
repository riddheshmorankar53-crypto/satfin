"""Fusion/refinement: a small U-Net that predicts a residual on top of the mask-blended warp."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from satfin.models.ifnet import conv
from satfin.models.warp import backward_warp


class ContextNet(nn.Module):
    """Per-frame context features, warped to time t with the estimated flow."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(conv(channels, width), conv(width, width))

    def forward(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return backward_warp(self.net(img), flow)


class FusionUNet(nn.Module):
    """Input: warped frames, blend, mask, flow, warped context. Output: residual (B,C,H,W) in [-1,1].

    H and W must be divisible by 4.
    """

    def __init__(self, channels: int = 1, width: int = 32) -> None:
        super().__init__()
        self.context = ContextNet(channels, width // 2)
        cin = 3 * channels + 1 + 4 + width  # w0, w1, merged, mask, flow, ctx0+ctx1
        self.e1 = nn.Sequential(conv(cin, width), conv(width, width))
        self.e2 = nn.Sequential(conv(width, 2 * width, 2), conv(2 * width, 2 * width))
        self.e3 = nn.Sequential(conv(2 * width, 4 * width, 2), conv(4 * width, 4 * width))
        self.d2 = nn.Sequential(conv(4 * width + 2 * width, 2 * width), conv(2 * width, 2 * width))
        self.d1 = nn.Sequential(conv(2 * width + width, width), conv(width, width))
        self.out = nn.Conv2d(width, channels, 3, 1, 1)

    def forward(self, I0: torch.Tensor, I1: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor,
                merged: torch.Tensor, mask: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        ctx = torch.cat([self.context(I0, flow[:, :2]), self.context(I1, flow[:, 2:])], 1)
        x1 = self.e1(torch.cat([w0, w1, merged, mask, flow, ctx], 1))
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.d2(torch.cat([up(x3, x2), x2], 1))
        y1 = self.d1(torch.cat([up(y2, x1), x1], 1))
        return torch.tanh(self.out(y1))
