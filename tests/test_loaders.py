import struct
from datetime import datetime

import h5py
import numpy as np
import pytest

from satfin.data.download import himawari_time
from satfin.data.loaders import group_scans
from satfin.data.loaders.himawari import read_himawari_bt
from satfin.data.loaders.insat import parse_time, read_insat_bt
from satfin.data.preprocess import align


def test_insat_lut_and_fill(tmp_path):
    p = tmp_path / "3SIMG_01JUN2025_0015_L1B_STD_V01R00.h5"
    counts = np.array([[[0, 10, 1023], [1023, 5, 0]]], np.uint16)
    with h5py.File(p, "w") as f:
        d = f.create_dataset("IMG_TIR1", data=counts)
        d.attrs["_FillValue"] = np.array([1023], np.uint16)
        f.create_dataset("IMG_TIR1_TEMP", data=np.linspace(180, 330, 1024).astype(np.float32))
        f.attrs["Acquisition_Start_Time"] = b"01-JUN-2025T00:15:05"
    bt, meta = read_insat_bt(p)
    assert bt.shape == (2, 3) and np.isnan(bt[0, 2]) and np.isnan(bt[1, 0])
    assert bt[0, 0] == pytest.approx(180) and bt[0, 1] == pytest.approx(180 + 150 * 10 / 1023)
    assert meta["time"] == datetime(2025, 6, 1, 0, 15, 5)


def test_insat_time_from_filename():
    assert parse_time("3SIMG_01JUN2025_0015_L1B_STD_V01R00.h5") == datetime(2025, 6, 1, 0, 15)


def _hsd(path, seg, counts, mjd=60827.125, coff=180.5):
    """Minimal HSD file with the header fields the reader uses (layout checked against real Himawari-9 files)."""
    lines, cols = counts.shape
    b = {n: bytearray(ln) for n, ln in {1: 282, 2: 50, 3: 127, 4: 139, 5: 147, 6: 259, 7: 47,
                                        8: 81, 9: 75, 10: 47, 11: 259}.items()}
    for n, blk in b.items():
        blk[0] = n
        struct.pack_into("<H", blk, 1, len(blk))
    b[1][6:16] = b"Himawari-9"
    b[1][38:42] = b"R301"
    struct.pack_into("<d", b[1], 46, mjd)
    struct.pack_into("<I", b[1], 70, sum(map(len, b.values())))
    struct.pack_into("<HHH", b[2], 3, 16, cols, lines)
    struct.pack_into("<dIIff", b[3], 3, 140.7, 20466275, 20466275, coff, 1640.5)
    c, h, k = 2.99792458e8, 6.62606957e-34, 1.3806488e-23
    struct.pack_into("<HdHHHdd", b[5], 3, 13, 10.4, 12, 65535, 65534, 0.01, -1.0)
    struct.pack_into("<9d", b[5], 35, 0.0, 1.0, 0.0, 0, 1, 0, c, h, k)
    b[7][3], b[7][4] = 2, seg
    path.write_bytes(b"".join(b[n] for n in range(1, 12)) + counts.astype("<u2").tobytes())


def test_himawari_segments_calibration_and_invalid(tmp_path):
    counts = np.full((3, 4), 800, np.uint16)
    counts[0, 0] = 65535
    _hsd(tmp_path / "HS_H09_20250601_0300_B13_R301_R20_S0202.DAT", 2, counts)
    _hsd(tmp_path / "HS_H09_20250601_0300_B13_R301_R20_S0102.DAT", 1, counts * 0 + 900)
    (scan,) = group_scans(list(tmp_path.iterdir()))
    bt, meta = read_himawari_bt(scan)
    assert bt.shape == (6, 4) and np.isnan(bt[3, 0])
    # counts 900 (seg 1) / 800 (seg 2) -> radiance 8 / 7 W m-2 sr-1 um-1; Planck forward must give them back
    c, h, k, lam = 2.99792458e8, 6.62606957e-34, 1.3806488e-23, 10.4e-6
    planck = lambda T: 2 * h * c ** 2 / lam ** 5 / (np.exp(h * c / (lam * k * T)) - 1) / 1e6
    assert planck(float(bt[0, 0])) == pytest.approx(8.0, rel=1e-5)
    assert planck(float(bt[4, 1])) == pytest.approx(7.0, rel=1e-5)
    assert meta["time"] == datetime(2025, 6, 1, 3, 0) and meta["band"] == 13


def test_himawari_time_and_align():
    assert himawari_time("HS_H09_20250601_0300_B13_R303_R20_S0101.DAT.bz2") == datetime(2025, 6, 1, 3, 5)
    img = np.arange(60, dtype=np.float32).reshape(6, 10)
    shifted = np.roll(img, 3, axis=1)  # same scene, window moved so content sits 3 columns right (coff + 3)
    m = [{"projection": {"coff": 100.5, "loff": 50.5}}, {"projection": {"coff": 103.5, "loff": 50.5}}]
    a, b = align([img, shifted], m)
    np.testing.assert_array_equal(a, b)
