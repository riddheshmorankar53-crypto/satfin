"""GOES ABI L1b reader (netCDF4 directly: xarray takes ~5 s per ABI file to open)."""
from pathlib import Path

import netCDF4
import numpy as np

from satfin.data.download import parse_start_time


def rad_to_bt(rad: np.ndarray, fk1: float, fk2: float, bc1: float, bc2: float) -> np.ndarray:
    """Radiance (mW m-2 sr-1 (cm-1)-1) to brightness temperature (K), GOES-R PUG inverse Planck."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (fk2 / np.log(fk1 / rad + 1.0) - bc1) / bc2


def read_goes_bt(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read an ABI L1b IR file (bands 7-16) as BT in K (float32, NaN where invalid) plus metadata."""
    with netCDF4.Dataset(path) as ds:
        band = int(ds["band_id"][:].item())
        if band < 7:
            raise ValueError(f"band {band} is reflective; BT only exists for bands 7-16")
        rad = ds["Rad"][:]  # masked array, scale/offset applied
        dqf = ds["DQF"][:]
        k = {n: float(ds[f"planck_{n}"][:]) for n in ("fk1", "fk2", "bc1", "bc2")}
        e = ds["geospatial_lat_lon_extent"]
        meta = {
            "time": parse_start_time(Path(path).name),
            "platform": ds.platform_ID,
            "scene": ds.scene_id,
            "band": band,
            "center": (round(float(e.geospatial_lat_center), 2), round(float(e.geospatial_lon_center), 2)),
        }
    bt = np.ma.filled(rad_to_bt(rad, **k).astype(np.float32), np.nan)
    bt[np.ma.filled(dqf, 3) >= 2] = np.nan  # DQF 0 good, 1 conditionally usable, >=2 bad/no value
    bt[~np.isfinite(bt)] = np.nan
    return bt, meta
