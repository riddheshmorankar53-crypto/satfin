"""Insert K frames between every pair of a frame sequence at any cadence; tiled for large scenes.

python -m satfin.infer --ckpt runs/gpu/best.pt --seq data/processed/<name> --every 10 --k 9 --range 0:31
python -m satfin.infer --ckpt runs/gpu/best.pt --files "data/raw/himawari9/FLDK/C13/*.DAT.bz2" --k 1
  --seq     processed folder (frames.npy/times.npy/meta.json); --every N keeps every Nth frame
            (simulated low cadence) and, when N == k + 1, the skipped frames are ground truth
  --files   raw GOES .nc / Himawari .DAT(.bz2) / INSAT .h5 files, any cadence
  30 -> 15 -> 7.5 min: --k 3 (t = 1/4, 2/4, 3/4); 10 -> 1 min: --k 9

Writes to --out: interpolated.nc (BT in K, time, `interpolated` flag, source metadata), interp_*.png,
interpolated.gif/.mp4, original.gif, comparison.gif/.mp4 (original held vs SatFIN [vs ground truth]).
"""
import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import netCDF4
import numpy as np
import torch

from satfin.config import load_config
from satfin.data.loaders import group_scans, read_scan
from satfin.data.preprocess import align, denormalize, normalize, same_place
from satfin.env_check import get_device
from satfin.models.satfin import build_model
from satfin.visualize import preview, save_gif, save_mp4, side_by_side, to_u8


def load_model(ckpt_path: str | Path, dev: torch.device | None = None):
    """Model (eval mode) and its training config from a checkpoint."""
    dev = dev or get_device()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = build_model(ckpt["cfg"]["model"]).to(dev).eval()
    model.load_state_dict(ckpt["model"])
    return model, ckpt["cfg"]


def _ramp(n: int, overlap: int) -> np.ndarray:
    """1-D blending weight: linear ramps of length `overlap` at both ends, 1 in the middle."""
    r = np.ones(n, np.float32)
    if overlap:
        up = (np.arange(overlap, dtype=np.float32) + 1) / (overlap + 1)
        r[:overlap] = np.minimum(r[:overlap], up)
        r[-overlap:] = np.minimum(r[-overlap:], up[::-1])
    return r


def _starts(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, stride))
    return s + [n - tile]


@torch.no_grad()
def interpolate(model, i0: np.ndarray, i1: np.ndarray, ts: list[float], tile: int = 1024,
                overlap: int = 128, batch: int = 8) -> list[np.ndarray]:
    """HxW frames in [0,1] -> one predicted HxW frame per t (any t in [0,1]).

    Frames larger than `tile` are split into overlapping tiles whose predictions are blended with
    linear ramps over the overlap, so no seams appear.
    """
    dev = next(model.parameters()).device
    H, W = i0.shape
    a, b = (torch.from_numpy(np.asarray(x, np.float32))[None, None].to(dev) for x in (i0, i1))
    acc = np.zeros((len(ts), H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    th, tw = min(tile, H), min(tile, W)
    for y in _starts(H, tile, tile - overlap):
        for x in _starts(W, tile, tile - overlap):
            wt = np.outer(_ramp(th, overlap if th < H else 0), _ramp(tw, overlap if tw < W else 0))
            pa, pb = a[..., y:y + th, x:x + tw], b[..., y:y + th, x:x + tw]
            for j in range(0, len(ts), batch):
                tt = torch.tensor(ts[j:j + batch], device=dev, dtype=torch.float32)[:, None]
                n = len(tt)
                pred = model(pa.expand(n, -1, -1, -1), pb.expand(n, -1, -1, -1), tt)["pred"][:, 0]
                acc[j:j + n, y:y + th, x:x + tw] += pred.float().cpu().numpy() * wt
            wsum[y:y + th, x:x + tw] += wt
    return list(acc / wsum)


def interpolate_sequence(model, frames: list[np.ndarray], times: np.ndarray, k: int,
                         **kw) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Insert k frames (t = j/(k+1)) between every consecutive pair.

    Returns all frames, their times (datetime64[s], linear in t) and a bool mask of interpolated frames.
    """
    ts = [(j + 1) / (k + 1) for j in range(k)]
    out, out_t, flag = [frames[0]], [times[0]], [False]
    for (f0, t0), (f1, t1) in zip(zip(frames, times), zip(frames[1:], times[1:])):
        out += interpolate(model, f0, f1, ts, **kw) + [f1]
        out_t += [t0 + (t1 - t0) * t for t in ts] + [t1]
        flag += [True] * k + [False]
    return out, np.array(out_t, "datetime64[s]"), np.array(flag)


def save_netcdf(path: Path, frames: list[np.ndarray], times: np.ndarray, interpolated: np.ndarray,
                bounds: tuple[float, float], meta: dict) -> None:
    """CF-style NetCDF: bt(time, y, x) in K, time in s since 1970, interpolated flag, source metadata attrs."""
    T, (H, W) = len(frames), frames[0].shape
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", T)
        ds.createDimension("y", H)
        ds.createDimension("x", W)
        tv = ds.createVariable("time", "i8", ("time",))
        tv.units, tv.standard_name = "seconds since 1970-01-01 00:00:00", "time"
        tv[:] = times.astype("datetime64[s]").astype("int64")
        fv = ds.createVariable("interpolated", "u1", ("time",))
        fv.long_name = "1 = frame generated by SatFIN, 0 = original observation"
        fv[:] = interpolated.astype("u1")
        bt = ds.createVariable("bt", "f4", ("time", "y", "x"), zlib=True, complevel=4, chunksizes=(1, H, W))
        bt.units, bt.long_name = "K", "brightness temperature"
        bt.valid_range = np.array(bounds, "f4")
        for i, f in enumerate(frames):
            bt[i, :, :] = denormalize(np.asarray(f, np.float32), *bounds)
        ds.title = "SatFIN temporally interpolated brightness temperature"
        ds.history = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} satfin.infer"
        for key, v in meta.items():
            ds.setncattr(f"source_{key}", v if isinstance(v, (int, float, str)) else json.dumps(v, default=str))


def load_frames(a: argparse.Namespace, dc: dict) -> tuple[list[np.ndarray], np.ndarray, dict, list | None]:
    """Normalized frames, times, metadata and (if available) the 1-min ground-truth frames per pair."""
    if a.seq:
        frames = np.load(a.seq / "frames.npy", mmap_mode="r")
        times = np.load(a.seq / "times.npy")
        lo, hi = (int(s) if s else None for s in a.range.split(":")) if a.range else (None, None)
        idx = np.arange(len(frames))[lo:hi]
        keep = idx[::a.every]
        gt = None
        if a.every == a.k + 1 and a.every > 1:
            gt = [frames[i:j + 1].astype(np.float32) for i, j in zip(keep, keep[1:])]
        meta = json.loads((a.seq / "meta.json").read_text())
        return [frames[i].astype(np.float32) for i in keep], times[keep], meta, gt
    files = sorted(Path(p) for p in glob.glob(a.files))
    if not files:
        raise SystemExit(f"no files match {a.files}")
    scans = sorted((read_scan(g, dc.get("insat")) for g in group_scans(files)), key=lambda s: s[1]["time"])
    if not all(same_place(scans[0][1], m) for _, m in scans):
        raise SystemExit("input files cover different scene positions; pass one sector at a time")
    frames = []
    for bt in align([b for b, _ in scans], [m for _, m in scans]):
        bt = np.where(np.isnan(bt), np.nanmean(bt), bt)
        frames.append(normalize(bt, dc["bt_min"], dc["bt_max"]).astype(np.float32))
    m = scans[0][1]
    meta = {k: v for k, v in m.items() if k != "time"}
    return frames, np.array([s[1]["time"] for s in scans], "datetime64[s]"), meta, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", type=Path, help="processed sequence folder")
    src.add_argument("--files", help="glob of raw satellite files (quote it)")
    p.add_argument("--k", type=int, default=9, help="frames to insert between each pair (t = j/(k+1))")
    p.add_argument("--every", type=int, default=1, help="--seq only: keep every Nth frame as input")
    p.add_argument("--range", default=None, help="--seq only: frame slice start:stop, e.g. 0:31")
    p.add_argument("--tile", type=int, default=1024, help="tile size for large scenes")
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    a = p.parse_args()

    model, cfg = load_model(a.ckpt)
    dc = {**load_config()["data"], **cfg["data"]}
    frames, times, meta, gt = load_frames(a, dc)
    if len(frames) < 2:
        raise SystemExit("need at least 2 input frames")
    seq, seq_t, flag = interpolate_sequence(model, frames, times, a.k, tile=a.tile, overlap=a.overlap)

    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("interp_*.png"):
        old.unlink()
    save_netcdf(a.out / "interpolated.nc", seq, seq_t, flag, (dc["bt_min"], dc["bt_max"]), meta)
    for j, f in enumerate(seq):
        imageio.imwrite(a.out / f"interp_{j:03d}.png", to_u8(f))
    small = [preview(f) for f in seq]  # animations of large scenes are downscaled to <= 1024 px
    save_gif(small, a.out / "interpolated.gif", fps=a.fps)
    save_mp4(small, a.out / "interpolated.mp4", fps=a.fps)
    save_gif(small[::a.k + 1], a.out / "original.gif", fps=a.fps / (a.k + 1))
    held = [small[i // (a.k + 1) * (a.k + 1)] for i in range(len(seq))]  # last real frame
    cols = {"original": held, "SatFIN": small}
    if gt is not None:
        cols["ground truth"] = [preview(f) for g in gt for f in g[:-1]] + [preview(gt[-1][-1])]
        save_gif(cols["ground truth"], a.out / "ground_truth.gif", fps=a.fps)
    labels = [f"{str(t)[11:19]}{' *' if f else ''}" for t, f in zip(seq_t, flag)]
    comp = side_by_side(cols, labels)
    save_gif(comp, a.out / "comparison.gif", fps=a.fps)
    save_mp4(comp, a.out / "comparison.mp4", fps=a.fps)
    print(f"{len(frames)} input -> {len(seq)} frames ({flag.sum()} interpolated), {seq[0].shape} -> {a.out}/ "
          f"interpolated.nc, interp_*.png, interpolated.gif/.mp4, original.gif, comparison.gif/.mp4"
          + (", ground_truth.gif" if gt is not None else ""))


if __name__ == "__main__":
    main()
