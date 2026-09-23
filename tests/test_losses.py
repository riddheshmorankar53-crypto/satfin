import numpy as np
import torch

from satfin.baselines import flow_interp, linear_blend
from satfin.losses import charbonnier, convective_weight, total_loss
from satfin.models.satfin import SatFIN

CFG = dict(charbonnier_eps=1e-3, cold_bt=235.0, cold_weight=4.0, grad_thresh_K=5.0, grad_weight=2.0,
           smooth_weight=0.01, edge_weight=0.1)
BOUNDS = (180.0, 330.0)


def test_charbonnier_zero_for_identical():
    x = torch.rand(2, 1, 8, 8)
    assert charbonnier(x, x).item() < 2e-3


def test_convective_weight():
    warm = torch.full((1, 1, 2, 2), (300 - 180) / 150)
    cold = warm.clone()
    cold[..., 0, 0] = (220 - 180) / 150  # cold cloud top in It only
    w = convective_weight(warm, cold, warm, BOUNDS, CFG)
    assert w[..., 0, 0] == 5 and w[..., 1, 1] == 1
    fast = warm.clone()
    fast[..., 1, 1] += 10 / 150  # +10 K between I0 and I1
    assert convective_weight(warm, warm, fast, BOUNDS, CFG)[..., 1, 1] == 3


def test_total_loss_backprops():
    model = SatFIN(1, (16, 12, 8), (4, 2, 1), 8)
    b = {k: torch.rand(2, 1, 32, 32) for k in ("I0", "It", "I1")}
    b["t"] = torch.tensor([[0.5], [0.2]])
    loss, parts = total_loss(model(b["I0"], b["I1"], b["t"]), b, BOUNDS, CFG)
    loss.backward()
    assert set(parts) == {"rec", "smooth", "edge"} and loss.item() > 0


def _blob(cx: float) -> np.ndarray:
    ys, xs = np.mgrid[:64, :64]
    return np.exp(-((xs - cx) ** 2 + (ys - 32) ** 2) / 50).astype(np.float32)


def test_baselines():
    i0, i1, gt = _blob(28), _blob(36), _blob(32)  # 8 px; Farneback loses a 16 px shift of a smooth blob
    assert np.allclose(linear_blend(i0, i1, 0.0), i0)
    err_lin = np.abs(linear_blend(i0, i1, 0.5) - gt).mean()
    for m in ("dis", "farneback"):
        assert np.abs(flow_interp(i0, i1, 0.5, m) - gt).mean() < err_lin  # motion beats blending
