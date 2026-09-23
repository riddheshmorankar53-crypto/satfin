"""Download GOES ABI L1b radiance files from the public NOAA AWS buckets (anonymous).

python -m satfin.data.download --satellite goes19 --band 13 --start 2025-06-01T18:00 --hours 3 --sector M1
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import s3fs
from tqdm import tqdm

from satfin.config import load_config


def parse_start_time(filename: str) -> datetime:
    """Scan start time from an ABI filename, e.g. '..._s20251521800279_...' -> 2025-06-01 18:00:27."""
    s = Path(filename).name.split("_s")[1][:13]  # YYYYDDDHHMMSS (drop tenths)
    return datetime.strptime(s, "%Y%j%H%M%S")


def list_files(fs: s3fs.S3FileSystem, satellite: str, sector: str, band: int,
               start: datetime, hours: float) -> list[str]:
    """List S3 keys for one sector (F, C, M1, M2) and band whose scan start is in [start, start + hours)."""
    end = start + timedelta(hours=hours)
    product = f"ABI-L1b-Rad{sector[0]}"
    tag = f"-Rad{sector}-"  # e.g. '-RadM1-'
    band_tag = f"C{band:02d}_"
    keys, hour = [], start.replace(minute=0, second=0, microsecond=0)
    while hour < end:
        prefix = f"noaa-{satellite}/{product}/{hour:%Y/%j/%H}"
        try:
            listing = fs.ls(prefix)
        except FileNotFoundError:
            listing = []
        keys += [k for k in listing if tag in k and band_tag in k and start <= parse_start_time(k) < end]
        hour += timedelta(hours=1)
    return sorted(keys, key=parse_start_time)


def download(keys: list[str], fs: s3fs.S3FileSystem, out_dir: Path) -> list[Path]:
    """Download keys into out_dir, skipping files already present with the right size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in tqdm(keys, desc="download", unit="file"):
        dst = out_dir / Path(key).name
        if not (dst.exists() and dst.stat().st_size == fs.info(key)["size"]):
            tmp = dst.with_suffix(".part")
            fs.get(key, str(tmp))
            tmp.replace(dst)  # atomic: never leave a truncated .nc behind
        paths.append(dst)
    return paths


def main() -> None:
    cfg = load_config()["data"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--satellite", default=cfg["satellite"])
    p.add_argument("--sector", default=cfg["sector"], help="F | C | M1 | M2")
    p.add_argument("--band", type=int, default=cfg["band"])
    p.add_argument("--start", required=True, type=datetime.fromisoformat, help="UTC, e.g. 2025-06-01T18:00")
    p.add_argument("--hours", type=float, default=3)
    p.add_argument("--out", type=Path, default=None, help="default: <raw_dir>/<satellite>/<sector>/C<band>")
    a = p.parse_args()

    out = a.out or Path(cfg["raw_dir"]) / a.satellite / a.sector / f"C{a.band:02d}"
    fs = s3fs.S3FileSystem(anon=True)
    keys = list_files(fs, a.satellite, a.sector, a.band, a.start, a.hours)
    print(f"{len(keys)} files for {a.satellite} {a.sector} C{a.band:02d} from {a.start:%Y-%m-%d %H:%M} (+{a.hours}h) -> {out}")
    if keys:
        download(keys, fs, out)
        t = [parse_start_time(k) for k in keys]
        print(f"first {t[0]}  last {t[-1]}")


if __name__ == "__main__":
    main()
