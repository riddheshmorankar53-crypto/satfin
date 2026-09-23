import pytest
import torch

from satfin.models.satfin import SatFIN
from satfin.models.warp import backward_warp


def test_warp_zero_flow_is_identity():
    img = torch.rand(2, 3, 17, 23)
    assert torch.allclose(backward_warp(img, torch.zeros(2, 2, 17, 23)), img, atol=1e-5)


def test_warp_integer_shift():
    img = torch.rand(1, 1, 8, 10)
    flow = torch.zeros(1, 2, 8, 10)
    flow[:, 0] = 2  # sample 2 px to the right
    out = backward_warp(img, flow)
    assert torch.allclose(out[..., :-2], img[..., 2:], atol=1e-5)


@pytest.mark.parametrize("channels", [1, 3])
def test_satfin_forward_any_size(channels):
    model = SatFIN(channels, (16, 12, 8), (4, 2, 1), 8)
    x0, x1 = torch.rand(2, channels, 50, 70), torch.rand(2, channels, 50, 70)
    out = model(x0, x1, torch.tensor([[0.3], [0.7]]))
    assert out["pred"].shape == (2, channels, 50, 70)
    assert out["flow"].shape == (2, 4, 50, 70) and out["mask"].shape == (2, 1, 50, 70)
    assert 0 <= out["pred"].min() and out["pred"].max() <= 1
    out["pred"].mean().backward()
    assert all(p.grad is not None for p in model.parameters())
