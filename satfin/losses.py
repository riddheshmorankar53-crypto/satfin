"""Training losses. Images are normalized BT in [0,1]; `bounds` = (bt_min, bt_max) in K."""
import torch


def charbonnier(pred: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor | None = None,
                eps: float = 1e-3) -> torch.Tensor:
    """Weighted Charbonnier (smooth L1) loss."""
    err = torch.sqrt((pred - gt) ** 2 + eps ** 2)
    if weight is None:
        return err.mean()
    return (err * weight).sum() / weight.expand_as(err).sum()


def convective_weight(I0: torch.Tensor, It: torch.Tensor, I1: torch.Tensor, bounds: tuple[float, float],
                      cfg: dict) -> torch.Tensor:
    """1 + cold_weight where any frame is colder than cold_bt, + grad_weight where |dBT| across the gap is large."""
    span = bounds[1] - bounds[0]
    cold_norm = (cfg["cold_bt"] - bounds[0]) / span
    cold = (torch.minimum(torch.minimum(I0, It), I1) < cold_norm).float()
    fast = ((I1 - I0).abs() * span > cfg["grad_thresh_K"]).float()
    return 1 + cfg["cold_weight"] * cold + cfg["grad_weight"] * fast


def _grads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    """Mean absolute first-order flow gradient (pixels)."""
    dx, dy = _grads(flow)
    return dx.abs().mean() + dy.abs().mean()


def edge_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 between image gradients, for edge sharpness."""
    (px, py), (gx, gy) = _grads(pred), _grads(gt)
    return (px - gx).abs().mean() + (py - gy).abs().mean()


def total_loss(out: dict, batch: dict, bounds: tuple[float, float], cfg: dict) -> tuple[torch.Tensor, dict]:
    """Combined training loss and its parts (floats, for logging)."""
    w = convective_weight(batch["I0"], batch["It"], batch["I1"], bounds, cfg)
    parts = {
        "rec": charbonnier(out["pred"], batch["It"], w, cfg["charbonnier_eps"]),
        "smooth": cfg["smooth_weight"] * flow_smoothness(out["flow"]),
    }
    if cfg["edge_weight"]:
        parts["edge"] = cfg["edge_weight"] * edge_loss(out["pred"], batch["It"])
    loss = sum(parts.values())
    return loss, {k: v.item() for k, v in parts.items()}
