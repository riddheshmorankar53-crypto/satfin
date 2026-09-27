"""Raw satellite files (GOES .nc, Himawari .DAT.bz2) -> normalized BT sequences.

python -m satfin.data.preprocess            # every folder of satellite files under data.raw_dir

Each contiguous sequence (same sector position and shape, no gap > 1.5x the median cadence) is saved as
<processed_dir>/<name>/frames.npy (T,H,W float16 in [0,1]), times.npy (datetime64[s]), meta.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from satfin.config import load_config
from satfin.data.loaders import PATTERNS, find_files, group_scans, read_scan


def normalize(bt: np.ndarray, bt_min: float, bt_max: float) -> np.ndarray:
    """Map BT (K) to [0,1] with fixed physical bounds, clipping outside."""
    return np.clip((bt - bt_min) / (bt_max - bt_min), 0.0, 1.0)


def denormalize(x: np.ndarray, bt_min: float, bt_max: float) -> np.ndarray:
    """Inverse of normalize (for values inside the bounds)."""
    return x * (bt_max - bt_min) + bt_min


MAX_JITTER_PX = 20  # Himawari target-area windows shift by a few px between scans; more means a new target


def same_place(a: dict, b: dict) -> bool:
    """Same scene position: equal centers, or (fixed-grid projection) offsets within MAX_JITTER_PX."""
    if "projection" in a and "projection" in b:
        pa, pb = a["projection"], b["projection"]
        return (pa["sub_lon"] == pb["sub_lon"] and abs(pa["coff"] - pb["coff"]) <= MAX_JITTER_PX
                and abs(pa["loff"] - pb["loff"]) <= MAX_JITTER_PX)
    return a["center"] == b["center"]


def align(bts: list[np.ndarray], metas: list[dict], ref: dict | None = None,
          margin: int | None = None) -> list[np.ndarray]:
    """Crop frames so pixel (r, c) is the same fixed-grid location in all of them (no-op without projection).

    Shifts are relative to `ref` (default metas[0]); `margin` px are cropped per side (default: the largest shift),
    so a fixed margin gives the same output size for any shift up to `margin`.
    """
    if "projection" not in metas[0]:
        return bts
    r = (ref or metas[0])["projection"]
    dc = [round(m["projection"]["coff"] - r["coff"]) for m in metas]
    dl = [round(m["projection"]["loff"] - r["loff"]) for m in metas]
    mc, ml = (margin, margin) if margin is not None else (max(map(abs, dc)), max(map(abs, dl)))
    H, W = bts[0].shape
    return [b[ml + y:H - ml + y, mc + x:W - mc + x] for b, x, y in zip(bts, dc, dl)]


def split_sequences(metas: list[dict], shapes: list[tuple]) -> list[list[int]]:
    """Group frame indices (sorted by time) into runs with same center/shape and no gap > 1.5x median cadence."""
    dts = [(b["time"] - a["time"]).total_seconds() for a, b in zip(metas, metas[1:])]
    max_gap_s = 1.5 * float(np.median(dts)) if dts else 0.0
    runs: list[list[int]] = []
    for i, m in enumerate(metas):
        prev = runs[-1][-1] if runs else None
        if (prev is None or not same_place(m, metas[prev]) or shapes[i] != shapes[prev]
                or (m["time"] - metas[prev]["time"]).total_seconds() > max_gap_s):
            runs.append([])
        runs[-1].append(i)
    return runs


def process_dir(raw_dir: Path, out_root: Path, cfg: dict) -> list[Path]:
    """Convert one folder of satellite files into one or more saved sequences."""
    scans = group_scans(find_files(raw_dir))
    pairs = sorted((read_scan(f) for f in tqdm(scans, desc=raw_dir.name, unit="scan")),
                   key=lambda p: p[1]["time"])
    bts, metas = [p[0] for p in pairs], [p[1] for p in pairs]
    out = []
    for run in split_sequences(metas, [b.shape for b in bts]):
        if len(run) < 3:
            continue
        stack = np.stack(align([bts[i] for i in run], [metas[i] for i in run]))
        nan = np.isnan(stack)
        if nan.any():
            # ponytail: NaNs filled with the frame mean; fine for rare bad pixels, add a validity mask if fill fraction grows
            stack = np.where(nan, np.nanmean(stack, axis=(1, 2), keepdims=True), stack)
        m0, m1 = metas[run[0]], metas[run[-1]]
        name = f"{m0['platform']}_{m0['scene']}_C{m0['band']:02d}_{m0['time']:%Y%m%dT%H%M%S}".replace(" ", "").replace("-", "")
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "frames.npy", normalize(stack, cfg["bt_min"], cfg["bt_max"]).astype(np.float16))
        np.save(d / "times.npy", np.array([metas[i]["time"] for i in run], dtype="datetime64[s]"))
        meta = {
            "platform": m0["platform"], "scene": m0["scene"], "band": m0["band"], "center": m0["center"],
            "start": f"{m0['time']:%Y-%m-%dT%H:%M:%S}", "end": f"{m1['time']:%Y-%m-%dT%H:%M:%S}",
            "n_frames": len(run), "shape": list(stack.shape[1:]),
            "bt_min": cfg["bt_min"], "bt_max": cfg["bt_max"],
            "nan_filled_frac": float(nan.mean()), "bt_range_K": [float(stack.min()), float(stack.max())],
            **({"projection": m0["projection"]} if "projection" in m0 else {}),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        out.append(d)
        print(f"  {name}: {len(run)} frames {meta['shape']}, BT {stack.min():.1f}-{stack.max():.1f} K, "
              f"NaN filled {meta['nan_filled_frac']:.2%}")
    return out


def main() -> None:
    cfg = load_config()["data"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path(cfg["raw_dir"]))
    p.add_argument("--out", type=Path, default=Path(cfg["processed_dir"]))
    a = p.parse_args()
    for d in sorted({f.parent for pat in PATTERNS for f in a.raw.rglob(pat)}):
        process_dir(d, a.out, cfg)


if __name__ == "__main__":
    main()
