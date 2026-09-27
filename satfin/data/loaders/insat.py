"""INSAT-3D / 3DR / 3DS Imager L1B HDF5 reader (MOSDAC, https://mosdac.gov.in, registration required).

Download the "L1B Standard" imager product manually (e.g. 3SIMG_01JUN2025_0015_L1B_STD_V01R00.h5)
into data/raw/insat3ds/. Expected layout (MOSDAC INSAT-3D L1B format; names are in configs/default.yaml
under `insat` so they can be changed without code edits if a product version differs):

  IMG_TIR1          (1, H, W) uint16 raw counts, 10.8 um thermal IR (closest to ABI band 13 / AHI B13)
  IMG_TIR1_TEMP     (1024,) float  count -> brightness temperature (K) lookup table
  _FillValue        attribute on IMG_TIR1 (fill counts)
  Acquisition_Start_Time   root attribute, e.g. "01-JUN-2025T00:15:05"

Run `python -m satfin.data.loaders.insat <file.h5>` to print a file's datasets and attributes.
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

DEFAULT_NAMES = {"counts": "IMG_TIR1", "lut": "IMG_TIR1_TEMP", "time_attr": "Acquisition_Start_Time"}


def parse_time(s: str) -> datetime:
    """'01-JUN-2025T00:15:05' (attribute) or '3SIMG_01JUN2025_0015_...' (filename) -> datetime."""
    s = s.decode() if isinstance(s, bytes) else str(s)
    for fmt in ("%d-%b-%YT%H:%M:%S", "%d-%b-%YT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s.strip().title(), fmt)
        except ValueError:
            pass
    m = re.search(r"(\d{2}[A-Z]{3}\d{4})_(\d{4})", s.upper())
    if not m:
        raise ValueError(f"cannot parse time from {s!r}")
    return datetime.strptime(m[1].title() + m[2], "%d%b%Y%H%M")


def read_insat_bt(path: str | Path, names: dict | None = None) -> tuple[np.ndarray, dict]:
    """BT (K, float32, NaN at fill) and metadata from one INSAT imager L1B HDF5 file."""
    n = {**DEFAULT_NAMES, **(names or {})}
    with h5py.File(path, "r") as f:
        for key in ("counts", "lut"):
            if n[key] not in f:
                raise KeyError(f"{n[key]!r} not in {path}; datasets: {list(f)}. "
                               f"Set insat.{key} in the config to the right name.")
        ds = f[n["counts"]]
        counts = ds[()].squeeze()
        fill = ds.attrs.get("_FillValue")
        lut = f[n["lut"]][()].astype(np.float32)
        t = f.attrs.get(n["time_attr"])
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()
                 if np.ndim(v) == 0}
    bt = lut[np.clip(counts, 0, len(lut) - 1)]
    bad = counts >= len(lut)
    if fill is not None:
        bad |= counts == np.asarray(fill).ravel()[0]
    bt[bad | (bt <= 0)] = np.nan
    meta = {
        "time": parse_time(t if t is not None else Path(path).name),
        "platform": str(attrs.get("Satellite_Name", Path(path).name[:2])),
        "scene": "IMG",
        "band": n["counts"],
        "center": tuple(bt.shape),
        "attrs": attrs,
    }
    return bt, meta


if __name__ == "__main__":
    with h5py.File(sys.argv[1], "r") as f:
        f.visititems(lambda k, v: print(k, getattr(v, "shape", ""), getattr(v, "dtype", "")))
        for k, v in f.attrs.items():
            print(f"@{k} = {v}")
