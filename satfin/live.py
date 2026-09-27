"""Live SatFIN: watch a satellite feed on AWS and interpolate every new scan as it arrives.

python -m satfin.live --ckpt runs/gpu-20k/best.pt                 # Himawari-9 Target B13, k=4 (2.5 min -> 30 s), 6 h window
python -m satfin.live --ckpt runs/gpu-20k/best.pt --once          # one poll, then exit
python -m satfin.live --ckpt runs/gpu-20k/best.pt --satellite goes19 --sector M1 --k 9

Raw files go to a temp dir, are read, then deleted; only the previous frame is kept (in memory).
Output in --out, pruned to the last --window-hours:
  frames/<UTC time>_{obs|int}.nc/.png   one frame each: obs = observed scan, int = SatFIN-interpolated
  state.json                            latest scan, lag, counters, last error (read by the dashboard Live view)
Interpolated frames lag real time by one scan interval: frames between two scans exist once the second arrives.
"""
import argparse
import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import s3fs

from satfin.data.download import himawari_time, list_files, list_himawari, parse_start_time
from satfin.data.loaders import read_scan
from satfin.data.loaders.himawari import scan_key
from satfin.data.preprocess import MAX_JITTER_PX, align, normalize, same_place
from satfin.infer import interpolate, load_model, save_netcdf
from satfin.visualize import to_u8

TIME_FMT = "%Y-%m-%dT%H%M%S"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def complete_scans(keys: list[str], hima: bool) -> list[tuple[datetime, list[str]]]:
    """(nominal time, keys) per scan, oldest first; Himawari scans only once all their segments are present."""
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(scan_key(k) if hima else k, []).append(k)
    out = []
    for g in groups.values():
        if hima and len(g) < int(Path(g[0]).name.split("_S")[1][2:4]):  # _S0110 = segment 01 of 10
            continue
        out.append(((himawari_time if hima else parse_start_time)(g[0]), sorted(g)))
    return sorted(out)


class LiveRunner:
    """Keeps the previous frame, interpolates each new one against it, writes and prunes outputs."""

    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.model, cfg = load_model(a.ckpt)
        self.bounds = (cfg["data"]["bt_min"], cfg["data"]["bt_max"])
        self.fs = s3fs.S3FileSystem(anon=True, use_listings_cache=False)  # cached listings hide new files
        self.hima = a.satellite.startswith("himawari")
        self.frames_dir = a.out / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.prev: tuple[np.ndarray, np.datetime64] | None = None
        self.ref: dict | None = None  # grid reference: all frames are cropped to its fixed-grid window
        self.last: datetime | None = None  # nominal time of the newest processed scan
        self.state = {"satellite": a.satellite, "sector": a.sector, "band": a.band, "k": a.k,
                      "window_hours": a.window_hours, "scans": 0, "frames_written": 0,
                      "latest_scan": None, "last_poll": None, "last_error": None}

    def poll(self) -> None:
        now = utcnow()
        start = self.last + timedelta(seconds=1) if self.last else now - timedelta(minutes=self.a.backfill_min)
        hours = (now - start).total_seconds() / 3600 + 0.2
        keys = (list_himawari if self.hima else list_files)(self.fs, self.a.satellite, self.a.sector,
                                                            self.a.band, start, hours)
        for nominal, group in complete_scans(keys, self.hima):
            self.last = nominal  # advance first: a bad scan is logged and skipped, not retried forever
            try:
                with tempfile.TemporaryDirectory() as d:
                    paths = [Path(d) / Path(k).name for k in group]
                    for k, p in zip(group, paths):
                        self.fs.get(k, str(p))
                    bt, meta = read_scan(paths)
            except (OSError, ValueError, KeyError) as e:
                self.state["last_error"] = f"{utcnow():%H:%M:%S} skipped {Path(group[0]).name}: {e}"
                print(self.state["last_error"], flush=True)
                continue
            self.add(bt, meta)
        self.prune(now)
        self.state["last_poll"] = f"{now:%Y-%m-%dT%H:%M:%S}"

    def add(self, bt: np.ndarray, meta: dict) -> None:
        """Interpolate against the previous frame (same place, gap <= max_gap_min) and write both."""
        if self.ref is None or not same_place(self.ref, meta):
            self.ref, self.prev = meta, None  # first frame or the target area moved: start over
        (bt,) = align([bt], [meta], ref=self.ref, margin=MAX_JITTER_PX)
        f = normalize(np.where(np.isnan(bt), np.nanmean(bt), bt), *self.bounds).astype(np.float32)
        t1 = np.datetime64(meta["time"], "s")
        info = {k: v for k, v in meta.items() if k != "time"}
        n = 0
        if self.prev is not None and 0 < (t1 - self.prev[1]) / np.timedelta64(60, "s") <= self.a.max_gap_min:
            f0, t0 = self.prev
            n = self.a.k
            ts = [(j + 1) / (self.a.k + 1) for j in range(self.a.k)]
            for t, p in zip(ts, interpolate(self.model, f0, f, ts)):
                self.write(p, t0 + (t1 - t0) * t, True, info)
        self.write(f, t1, False, info)
        self.prev = (f, t1)
        self.state["scans"] += 1
        self.state["latest_scan"] = str(t1)
        print(f"{utcnow():%H:%M:%S} scan {t1}: 1 observed + {n} interpolated frames", flush=True)

    def write(self, frame: np.ndarray, t: np.datetime64, interpolated: bool, info: dict) -> None:
        name = f"{t.astype(datetime):{TIME_FMT}}_{'int' if interpolated else 'obs'}"
        save_netcdf(self.frames_dir / f"{name}.nc", [frame], np.array([t]), np.array([interpolated]),
                    self.bounds, info)
        imageio.imwrite(self.frames_dir / f"{name}.png", to_u8(frame))
        self.state["frames_written"] += 1

    def prune(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=self.a.window_hours)
        for p in self.frames_dir.iterdir():
            if datetime.strptime(p.name[:17], TIME_FMT) < cutoff:
                p.unlink()

    def save_state(self) -> None:
        s = dict(self.state)
        if s["latest_scan"]:
            s["lag_min"] = round((utcnow() - datetime.fromisoformat(s["latest_scan"])).total_seconds() / 60, 1)
        tmp = self.a.out / "state.json.tmp"
        tmp.write_text(json.dumps(s, indent=2))
        tmp.replace(self.a.out / "state.json")  # atomic: the dashboard never reads a half-written file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--satellite", default="himawari9")
    p.add_argument("--sector", default="Target", help="Himawari: Target | FLDK; GOES: M1 | M2 | C | F")
    p.add_argument("--band", type=int, default=13)
    p.add_argument("--k", type=int, default=4, help="frames inserted between scans (2.5 min / 5 = 30 s)")
    p.add_argument("--window-hours", type=float, default=6)
    p.add_argument("--poll", type=float, default=30, help="seconds between S3 listings")
    p.add_argument("--backfill-min", type=float, default=30, help="on start, also process scans from the last N min")
    p.add_argument("--max-gap-min", type=float, default=8, help="larger gaps (missed scans) are not interpolated")
    p.add_argument("--out", type=Path, default=Path("outputs/live"))
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    live = LiveRunner(a)
    print(f"watching {a.satellite} {a.sector} B{a.band:02d}, k={a.k}, window {a.window_hours} h -> {a.out}", flush=True)
    while True:
        try:
            live.poll()
        except Exception as e:  # network hiccups, a bad file: log, keep the loop alive
            live.state["last_error"] = f"{utcnow():%H:%M:%S} {type(e).__name__}: {e}"
            print(live.state["last_error"], flush=True)
        live.save_state()
        if a.once:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
