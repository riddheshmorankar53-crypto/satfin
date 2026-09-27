"""Himawari-8/9 AHI Standard Data (HSD) reader for IR bands 7-16.

Public data: s3://noaa-himawari9 (anonymous), e.g.
  AHI-L1b-FLDK/2025/06/01/0300/HS_H09_20250601_0300_B13_FLDK_R20_S0110.DAT.bz2   full disk, 10 segments, 10 min
  AHI-L1b-Target/2025/06/01/0300/HS_H09_20250601_0300_B13_R301_R20_S0101.DAT.bz2  target area 500x500, 2.5 min
Layout follows the JMA "Himawari Standard Data User's Guide" v1.3 (header blocks 1-11, then uint16 counts).
"""
import bz2
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def _read(path: str | Path) -> bytes:
    raw = Path(path).read_bytes()
    return bz2.decompress(raw) if str(path).endswith(".bz2") else raw


def _blocks(raw: bytes) -> dict[int, bytes]:
    """Header blocks 1-11 keyed by block number."""
    out, off = {}, 0
    while len(out) < 11:
        n, ln = raw[off], int(np.frombuffer(raw, "<u2", 1, off + 1)[0])
        out[n] = raw[off:off + ln]
        off += ln
    return out


def _f8(b: bytes, off: int, n: int = 1) -> np.ndarray:
    return np.frombuffer(b, "<f8", n, off)


def read_segment(path: str | Path) -> tuple[np.ndarray, dict]:
    """One HSD file -> BT (K, float32, NaN where invalid) and metadata."""
    raw = _read(path)
    b = _blocks(raw)
    b1, b2, b3, b5 = b[1], b[2], b[3], b[5]
    if b1[5] != 0:
        raise ValueError("big-endian HSD not supported")
    cols, lines = np.frombuffer(b2, "<u2", 2, 5)
    header_len = int(np.frombuffer(b1, "<u4", 1, 70)[0])
    counts = np.frombuffer(raw, "<u2", int(cols) * int(lines), header_len).reshape(int(lines), int(cols))

    band = int(np.frombuffer(b5, "<u2", 1, 3)[0])
    if band < 7:
        raise ValueError(f"band {band} is reflective; BT only exists for bands 7-16")
    wl_um = float(_f8(b5, 5)[0])
    err_count, out_count = np.frombuffer(b5, "<u2", 2, 15)
    gain, offset = _f8(b5, 19, 2)
    c0, c1, c2, _, _, _, c, h, k = _f8(b5, 35, 9)

    rad = counts * gain + offset  # W m-2 sr-1 um-1
    lam = wl_um * 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        te = (h * c / (k * lam)) / np.log(2 * h * c ** 2 / (lam ** 5 * rad * 1e6) + 1)
    bt = (c0 + c1 * te + c2 * te ** 2).astype(np.float32)
    bt[(counts == err_count) | (counts == out_count) | ~np.isfinite(bt)] = np.nan

    mjd = float(_f8(b1, 46)[0])
    sub_lon, = _f8(b3, 3)
    cfac, lfac = np.frombuffer(b3, "<u4", 2, 11)
    coff, loff = np.frombuffer(b3, "<f4", 2, 19)
    meta = {
        "time": datetime(1858, 11, 17) + timedelta(days=mjd),
        "platform": b1[6:22].split(b"\0")[0].decode(),
        "scene": b1[38:42].decode(),  # FLDK, JP01, R301 ...
        "band": band,
        "segment": int(b[7][4]),
        "projection": {"sub_lon": float(sub_lon), "cfac": int(cfac), "lfac": int(lfac),
                       "coff": float(coff), "loff": float(loff)},
    }
    return bt, meta


def read_himawari_bt(paths: str | Path | list) -> tuple[np.ndarray, dict]:
    """BT (K) and metadata from one HSD file or the segments of one scan (stacked by segment number)."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    segs = sorted((read_segment(p) for p in paths), key=lambda s: s[1]["segment"])
    meta = dict(segs[0][1])
    p = meta["projection"]  # R301..R304 are one target area scanned 4x; it moves when coff/loff change
    meta["center"] = (p["sub_lon"], p["coff"], p["loff"])  # grouping key for preprocess
    return np.concatenate([s[0] for s in segs]), meta


def scan_key(path: str | Path) -> str:
    """Files of one scan share this key (everything before the segment tag)."""
    return re.sub(r"_S\d{4}\.DAT(\.bz2)?$", "", Path(path).name)
