"""Interpolate K frames between two frames of a processed sequence; save PNGs and GIFs to outputs/.

python -m satfin.infer --ckpt runs/cpu-2k/best.pt --seq data/processed/<name> --i0 0 --i1 10 --k 9
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from satfin.env_check import get_device
from satfin.models.satfin import build_model


def load_model(ckpt_path: str | Path, dev: torch.device | None = None):
    """Model (eval mode) and its training config from a checkpoint."""
    dev = dev or get_device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = build_model(ckpt["cfg"]["model"]).to(dev).eval()
    model.load_state_dict(ckpt["model"])
    return model, ckpt["cfg"]


@torch.no_grad()
def interpolate(model, i0: np.ndarray, i1: np.ndarray, ts: list[float]) -> list[np.ndarray]:
    """HxW frames in [0,1] -> one predicted HxW frame per t."""
    dev = next(model.parameters()).device
    a, b = (torch.from_numpy(np.asarray(x, np.float32))[None, None].to(dev) for x in (i0, i1))
    return [model(a, b, torch.tensor([[t]], device=dev))["pred"][0, 0].cpu().numpy() for t in ts]


def to_u8(x: np.ndarray) -> np.ndarray:
    """Normalized BT to 8-bit, inverted so cold cloud tops are bright."""
    return ((1 - np.clip(x, 0, 1)) * 255).astype(np.uint8)


def save_gif(frames: list[np.ndarray], path: Path, fps: float = 5) -> None:
    imageio.mimsave(path, [to_u8(f) for f in frames], duration=1 / fps, loop=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", type=Path, required=True, help="processed sequence folder")
    p.add_argument("--i0", type=int, default=0)
    p.add_argument("--i1", type=int, default=10)
    p.add_argument("--k", type=int, default=9, help="frames to insert (evenly spaced t)")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    a = p.parse_args()

    frames = np.load(a.seq / "frames.npy", mmap_mode="r")
    i0, i1 = frames[a.i0].astype(np.float32), frames[a.i1].astype(np.float32)
    ts = [(j + 1) / (a.k + 1) for j in range(a.k)]
    preds = interpolate(load_model(a.ckpt)[0], i0, i1, ts)

    a.out.mkdir(parents=True, exist_ok=True)
    seq = [i0, *preds, i1]
    for j, f in enumerate(seq):
        imageio.imwrite(a.out / f"interp_{j:03d}.png", to_u8(f))
    save_gif(seq, a.out / "interpolated.gif", fps=a.k + 1)
    save_gif([i0, i1], a.out / "original.gif", fps=1)
    if a.i1 - a.i0 == a.k + 1:  # real 1-min frames exist for every t: save ground truth too
        save_gif([frames[i].astype(np.float32) for i in range(a.i0, a.i1 + 1)], a.out / "ground_truth.gif", fps=a.k + 1)
    print(f"{len(seq)} frames -> {a.out}/interp_*.png, interpolated.gif, original.gif")


if __name__ == "__main__":
    main()
