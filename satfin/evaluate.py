"""Evaluate SatFIN vs linear blending vs OpenCV DIS / Farneback flow on held-out truth, overall and per gap.

python -m satfin.evaluate --ckpt runs/gpu/best.pt --n 100              # test split, config gaps
python -m satfin.evaluate --ckpt runs/gpu/best.pt --gap 10             # 10-frame gaps only (9 intermediate)
python -m satfin.evaluate --ckpt runs/gpu/best.pt --dirs "data/processed/Himawari*" --gap 4 --out outputs/himawari/results.md

Writes <out>.md, <out>.json and error-map figures (<out dir>/error_maps/) for a few triplets per gap.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from satfin.baselines import flow_interp, linear_blend
from satfin.data.dataset import TripletDataset, split_dirs
from satfin.infer import interpolate, load_model
from satfin.metrics import scores
from satfin.models.satfin import count_params
from satfin.visualize import error_figure

KEYS = ("psnr", "ssim", "mae", "cold_mae")


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
    p.add_argument("--dirs", default=None, help="glob of processed sequence folders (overrides --split)")
    p.add_argument("--gap", type=int, nargs="+", default=None, help="gaps in frames (default: config data.gaps)")
    p.add_argument("--n", type=int, default=100, help="evenly spaced triplets to score per gap")
    p.add_argument("--crop", type=int, default=None, help="center crop (default: config data.crop; 0 = full frame)")
    p.add_argument("--figs", type=int, default=3, help="error-map figures per gap")
    p.add_argument("--out", type=Path, default=Path("outputs/results.md"))
    a = p.parse_args()

    model, cfg = load_model(a.ckpt)
    dc = cfg["data"]
    gaps = a.gap or dc["gaps"]
    crop = dc["crop"] if a.crop is None else (a.crop or 10 ** 6)
    span = dc["bt_max"] - dc["bt_min"]
    cold = (cfg["loss"]["cold_bt"] - dc["bt_min"]) / span
    dirs = sorted(Path(d) for d in glob.glob(a.dirs)) if a.dirs else split_dirs(dc["processed_dir"], dc["splits"], dc.get("split_platforms"))[a.split]
    if not dirs:
        raise SystemExit("no sequences to evaluate")
    methods = {"Linear blend": lambda i0, i1, t: linear_blend(i0, i1, t),
               "OpenCV Farneback flow": lambda i0, i1, t: flow_interp(i0, i1, t, "farneback"),
               "OpenCV DIS flow": lambda i0, i1, t: flow_interp(i0, i1, t, "dis"),
               "SatFIN": lambda i0, i1, t: interpolate(model, i0, i1, [t])[0]}
    fig_dir = a.out.parent / "error_maps"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = {m: [] for m in methods}
    secs = {m: 0.0 for m in methods}
    per_gap, n_total = {}, 0
    for g in gaps:
        ds = TripletDataset(dirs, [g], crop, train=False)
        if not len(ds):
            print(f"gap {g}: no triplets, skipped")
            continue
        idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).round().astype(int)
        fig_at = set(idx[np.linspace(0, len(idx) - 1, min(a.figs, len(idx))).round().astype(int)].tolist()) if a.figs else set()
        g_rows = {m: [] for m in methods}
        g_secs = {m: 0.0 for m in methods}
        for i in tqdm(idx, desc=f"gap {g}"):
            b = ds[int(i)]
            i0, it, i1, t = b["I0"][0].numpy(), b["It"][0].numpy(), b["I1"][0].numpy(), b["t"].item()
            preds = {}
            for m, f in methods.items():
                t0 = time.perf_counter()
                preds[m] = np.clip(f(i0, i1, t), 0, 1).astype(np.float32)
                g_secs[m] += time.perf_counter() - t0
                g_rows[m].append(scores(preds[m], it, span, cold))
            if int(i) in fig_at:
                error_figure(it, preds, span, fig_dir / f"gap{g:02d}_{int(i):05d}.png", title=f"gap {g}, t={t:.2f}")
        per_gap[g] = {m: summarize(g_rows[m], 1000 * g_secs[m] / len(idx)) for m in methods}
        for m in methods:
            rows[m] += g_rows[m]
            secs[m] += g_secs[m]
        n_total += len(idx)
    overall = {m: summarize(rows[m], 1000 * secs[m] / n_total) for m in methods}

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    src = f"`{a.dirs}`" if a.dirs else f"split `{a.split}`"
    crop_txt = "full frame" if crop >= 10 ** 6 else f"{crop}x{crop} center crop"
    meta = {"split": a.dirs or a.split, "ckpt": a.ckpt.as_posix(), "n_per_gap": a.n, "n_total": n_total,
            "gaps_min": gaps, "crop": crop_txt, "cold_bt_K": cfg["loss"]["cold_bt"],
            "params": count_params(model), "train_steps": ckpt["step"], "best_val_psnr": ckpt["best"],
            "device": str(next(model.parameters()).device), "sequences": [d.name for d in dirs]}
    lines = [f"{src.capitalize() if not a.dirs else 'Sequences ' + src}, {n_total} evenly spaced triplets "
             f"(up to {a.n} per gap, gaps {gaps} frames), {crop_txt}, checkpoint `{meta['ckpt']}` "
             f"({ckpt['step']} steps). Cold-cloud MAE: pixels with ground-truth BT < {meta['cold_bt_K']:.0f} K.",
             "", "**Overall**", ""]
    lines += md_table(overall)
    for g, res in per_gap.items():
        lines += ["", f"**Gap {g} frames**", ""] + md_table(res)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    a.out.with_suffix(".json").write_text(json.dumps(
        {"meta": meta, "overall": overall, "per_gap": {str(g): r for g, r in per_gap.items()}}, indent=2))
    print("\n".join(lines))
    print(f"error maps -> {fig_dir}")


if __name__ == "__main__":
    main()
