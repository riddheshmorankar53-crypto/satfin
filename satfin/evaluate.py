"""Evaluate SatFIN vs linear blending vs OpenCV DIS flow on the test split; write outputs/results.md.

python -m satfin.evaluate --ckpt runs/cpu-2k/best.pt --n 128
"""
import argparse
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm

from satfin.baselines import flow_interp, linear_blend
from satfin.data.dataset import TripletDataset, split_dirs
from satfin.infer import interpolate, load_model


def scores(pred: np.ndarray, gt: np.ndarray, span: float) -> tuple[float, float, float]:
    """PSNR (dB), SSIM, MAE (K) for HxW frames in [0,1]."""
    mse = max(float(((pred - gt) ** 2).mean()), 1e-12)
    return (-10 * np.log10(mse), structural_similarity(pred, gt, data_range=1.0),
            float(np.abs(pred - gt).mean()) * span)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=128, help="evenly spaced triplets to score")
    p.add_argument("--out", type=Path, default=Path("outputs/results.md"))
    a = p.parse_args()

    model, cfg = load_model(a.ckpt)
    dc = cfg["data"]
    span = dc["bt_max"] - dc["bt_min"]
    ds = TripletDataset(split_dirs(dc["processed_dir"], dc["splits"])[a.split], dc["gaps"], dc["crop"], train=False)
    idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).round().astype(int)

    methods = {"Linear blend": lambda i0, i1, t: linear_blend(i0, i1, t),
               "OpenCV DIS flow": lambda i0, i1, t: flow_interp(i0, i1, t, "dis"),
               "SatFIN": lambda i0, i1, t: interpolate(model, i0, i1, [t])[0]}
    res = {m: [] for m in methods}
    for i in tqdm(idx, desc="eval"):
        b = ds[int(i)]
        i0, it, i1, t = b["I0"][0].numpy(), b["It"][0].numpy(), b["I1"][0].numpy(), b["t"].item()
        for m, f in methods.items():
            res[m].append(scores(np.clip(f(i0, i1, t), 0, 1).astype(np.float32), it, span))

    lines = [f"Split `{a.split}`, {len(idx)} evenly spaced triplets (gaps {dc['gaps']} min), "
             f"{dc['crop']}x{dc['crop']} center crop, checkpoint `{a.ckpt.as_posix()}`.", "",
             "| method | PSNR (dB, higher=better) | SSIM (higher=better) | MAE (K, lower=better) |", "|---|---|---|---|"]
    for m, r in res.items():
        ps, ss, ma = np.mean(r, 0)
        lines.append(f"| {m} | {ps:.2f} | {ss:.4f} | {ma:.3f} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
