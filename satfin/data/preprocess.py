"""Raw ABI NetCDF -> normalized BT sequences.

python -m satfin.data.preprocess            # every folder of .nc files under data.raw_dir

Each contiguous sequence (same sector position and shape, no gap > max_gap_s) is saved as
<processed_dir>/<name>/frames.npy (T,H,W float16 in [0,1]), times.npy (datetime64[s]), meta.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from satfin.config import load_config
from satfin.data.loaders.goes import read_goes_bt


def normalize(bt: np.ndarray, bt_min: float, bt_max: float) -> np.ndarray:
    """Map BT (K) to [0,1] with fixed physical bounds, clipping outside."""
    return np.clip((bt - bt_min) / (bt_max - bt_min), 0.0, 1.0)


def denormalize(x: np.ndarray, bt_min: float, bt_max: float) -> np.ndarray:
    """Inverse of normalize (for values inside the bounds)."""
    return x * (bt_max - bt_min) + bt_min


def split_sequences(metas: list[dict], shapes: list[tuple], max_gap_s: float) -> list[list[int]]:
    """Group frame indices (sorted by time) into runs with same center/shape and no big time gap."""
    runs: list[list[int]] = []
    for i, m in enumerate(metas):
        prev = runs[-1][-1] if runs else None
        if (prev is None or m["center"] != metas[prev]["center"] or shapes[i] != shapes[prev]
                or (m["time"] - metas[prev]["time"]).total_seconds() > max_gap_s):
            runs.append([])
        runs[-1].append(i)
    return runs


def process_dir(raw_dir: Path, out_root: Path, cfg: dict) -> list[Path]:
    """Convert one folder of ABI files into one or more saved sequences."""
    files = sorted(raw_dir.glob("*.nc"))
    pairs = sorted((read_goes_bt(f) for f in tqdm(files, desc=raw_dir.name, unit="file")),
                   key=lambda p: p[1]["time"])
    bts, metas = [p[0] for p in pairs], [p[1] for p in pairs]
    out = []
    for run in split_sequences(metas, [b.shape for b in bts], cfg["max_gap_s"]):
        if len(run) < 3:
            continue
        stack = np.stack([bts[i] for i in run])
        nan = np.isnan(stack)
        if nan.any():
            # ponytail: NaNs filled with the frame mean; fine for rare bad pixels, add a validity mask if fill fraction grows
            stack = np.where(nan, np.nanmean(stack, axis=(1, 2), keepdims=True), stack)
        m0, m1 = metas[run[0]], metas[run[-1]]
        name = f"{m0['platform']}_{m0['scene']}_C{m0['band']:02d}_{m0['time']:%Y%m%dT%H%M%S}".replace(" ", "")
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
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
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
    for d in sorted({f.parent for f in a.raw.rglob("*.nc")}):
        process_dir(d, a.out, cfg)


if __name__ == "__main__":
    main()
