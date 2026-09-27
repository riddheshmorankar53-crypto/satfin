"""Format dispatch: GOES ABI NetCDF (.nc), Himawari HSD (.DAT/.DAT.bz2)."""
from pathlib import Path

import numpy as np

from satfin.data.loaders.goes import read_goes_bt
from satfin.data.loaders.himawari import read_himawari_bt, scan_key

PATTERNS = ("*.nc", "*.DAT", "*.DAT.bz2")


def find_files(folder: Path) -> list[Path]:
    """All supported satellite files directly inside folder."""
    return sorted(p for pat in PATTERNS for p in Path(folder).glob(pat))


def group_scans(paths: list[Path]) -> list[list[Path]]:
    """One list of files per scan: Himawari segments of the same scan are grouped, other formats are 1 file/scan."""
    groups: dict[str, list[Path]] = {}
    for p in sorted(map(Path, paths)):
        groups.setdefault(scan_key(p) if ".DAT" in p.name else p.name, []).append(p)
    return list(groups.values())


def read_scan(files: list[Path]) -> tuple[np.ndarray, dict]:
    """BT (K, NaN where invalid) and metadata (time, platform, scene, band, center) for one scan."""
    name = files[0].name
    if name.endswith(".nc"):
        return read_goes_bt(files[0])
    if ".DAT" in name:
        return read_himawari_bt(files)
    raise ValueError(f"unsupported file type: {name}")
