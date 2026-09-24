"""Evaluate SatFIN vs linear blending vs OpenCV DIS flow on the test split, overall and per gap.

python -m satfin.evaluate --ckpt runs/cpu-2k/best.pt --n 100     # -> outputs/results.md, outputs/results.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

from satfin.baselines import flow_interp, linear_blend
from satfin.data.dataset import TripletDataset, split_dirs
from satfin.infer import interpolate, load_model
from satfin.models.satfin import count_params

KEYS = ("psnr", "ssim", "mae", "cold_mae")


def scores(pred: np.ndarray, gt: np.ndarray, span: float, cold: float) -> tuple[float, float, float, float]:
    """PSNR (dB), SSIM, MAE (K), and MAE (K) on cold-cloud pixels (gt < cold, normalized; NaN if none)."""
    err = np.abs(pred - gt)
    mse = max(float((err ** 2).mean()), 1e-12)
    c = gt < cold
    return (-10 * np.log10(mse), structural_similarity(pred, gt, data_range=1.0),
            float(err.mean()) * span, float(err[c].mean()) * span if c.any() else float("nan"))


def summarize(rows: list[tuple], ms: float) -> dict:
    """Mean of score rows (NaN-aware) plus mean ms/frame."""
    return {**dict(zip(KEYS, np.nanmean(np.array(rows, dtype=float), 0).tolist())), "ms": ms}


def md_table(res: dict) -> list[str]:
    lines = ["| method | PSNR (dB, higher=better) | SSIM (higher=better) | MAE (K, lower=better) | cold-cloud MAE (K, lower=better) | ms/frame |", "|---|---|---|---|---|---|"]
    lines += [f"| {m} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['mae']:.3f} | {r['cold_mae']:.3f} | {r['ms']:.1f} |"
              for m, r in res.items()]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=100, help="evenly spaced triplets to score per gap")
    p.add_argument("--out", type=Path, default=Path("outputs/results.md"))
    a = p.parse_args()

    model, cfg = load_model(a.ckpt)
    dc = cfg["data"]
    span = dc["bt_max"] - dc["bt_min"]
    cold = (cfg["loss"]["cold_bt"] - dc["bt_min"]) / span
    dirs = split_dirs(dc["processed_dir"], dc["splits"])[a.split]
    methods = {"Linear blend": lambda i0, i1, t: linear_blend(i0, i1, t),
               "OpenCV DIS flow": lambda i0, i1, t: flow_interp(i0, i1, t, "dis"),
               "SatFIN": lambda i0, i1, t: interpolate(model, i0, i1, [t])[0]}

    rows = {m: [] for m in methods}
    secs = {m: 0.0 for m in methods}
    per_gap, n_total = {}, 0
    for g in dc["gaps"]:
        ds = TripletDataset(dirs, [g], dc["crop"], train=False)
        idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).round().astype(int)
        g_rows = {m: [] for m in methods}
        g_secs = {m: 0.0 for m in methods}
        for i in tqdm(idx, desc=f"gap {g}"):
            b = ds[int(i)]
            i0, it, i1, t = b["I0"][0].numpy(), b["It"][0].numpy(), b["I1"][0].numpy(), b["t"].item()
            for m, f in methods.items():
                t0 = time.perf_counter()
                pred = f(i0, i1, t)
                g_secs[m] += time.perf_counter() - t0
                g_rows[m].append(scores(np.clip(pred, 0, 1).astype(np.float32), it, span, cold))
        per_gap[g] = {m: summarize(g_rows[m], 1000 * g_secs[m] / len(idx)) for m in methods}
        for m in methods:
            rows[m] += g_rows[m]
            secs[m] += g_secs[m]
        n_total += len(idx)
    overall = {m: summarize(rows[m], 1000 * secs[m] / n_total) for m in methods}

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = {"split": a.split, "ckpt": a.ckpt.as_posix(), "n_per_gap": a.n, "n_total": n_total,
            "gaps_min": dc["gaps"], "crop": dc["crop"], "cold_bt_K": cfg["loss"]["cold_bt"],
            "params": count_params(model), "train_steps": ckpt["step"], "best_val_psnr": ckpt["best"],
            "device": str(next(model.parameters()).device)}
    lines = [f"Split `{a.split}`, {n_total} evenly spaced triplets (up to {a.n} per gap, gaps {dc['gaps']} min), "
             f"{dc['crop']}x{dc['crop']} center crop, checkpoint `{meta['ckpt']}`. "
             f"Cold-cloud MAE: pixels with ground-truth BT < {meta['cold_bt_K']:.0f} K.", "", "**Overall**", ""]
    lines += md_table(overall)
    for g, res in per_gap.items():
        lines += ["", f"**Gap {g} min**", ""] + md_table(res)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    a.out.with_suffix(".json").write_text(json.dumps(
        {"meta": meta, "overall": overall, "per_gap": {str(g): r for g, r in per_gap.items()}}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
