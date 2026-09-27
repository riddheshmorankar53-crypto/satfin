import json

import numpy as np
import pytest

from satfin.data.dataset import TripletDataset, split_dirs
from satfin.data.loaders.goes import rad_to_bt
from satfin.data.preprocess import denormalize, normalize

# GOES-19 band 13 constants
K = dict(fk1=10860.4, fk2=1395.19, bc1=0.07481, bc2=0.99975)


def test_rad_to_bt_inverts_planck():
    bt = np.array([200.0, 235.0, 300.0])
    rad = K["fk1"] / (np.exp(K["fk2"] / (K["bc1"] + K["bc2"] * bt)) - 1)
    np.testing.assert_allclose(rad_to_bt(rad, **K), bt, atol=1e-6)


def test_normalize_roundtrip_and_clip():
    bt = np.array([150.0, 180.0, 255.0, 330.0, 400.0])
    x = normalize(bt, 180, 330)
    assert x.min() == 0 and x.max() == 1
    np.testing.assert_allclose(denormalize(x[1:4], 180, 330), bt[1:4])


@pytest.fixture
def seq_dir(tmp_path):
    d = tmp_path / "seq"
    d.mkdir()
    T = 12
    np.save(d / "frames.npy", np.random.rand(T, 40, 48).astype(np.float16))
    times = np.datetime64("2025-06-01T18:00:00", "s") + np.arange(T) * np.timedelta64(60, "s")
    times[7] += np.timedelta64(3, "s")  # real cadence jitter
    np.save(d / "times.npy", times)
    (d / "meta.json").write_text(json.dumps({"start": "2025-06-01T18:00:00", "platform": "G19"}))
    return d


@pytest.mark.parametrize("train", [True, False])
def test_dataset_shapes_and_ranges(seq_dir, train):
    ds = TripletDataset([seq_dir], gaps=[2, 4, 10], crop=32, train=train)
    T = 12
    assert len(ds) == sum((T - g) * (g - 1) for g in (2, 4, 10))
    for i in range(0, len(ds), 7):
        s = ds[i]
        for k in ("I0", "It", "I1"):
            assert s[k].shape == (1, 32, 32) and s[k].dtype.is_floating_point
            assert 0 <= s[k].min() and s[k].max() <= 1
        assert s["t"].shape == (1,) and 0 < s["t"].item() < 1


def test_split_dirs_by_date(seq_dir):
    s = split_dirs(seq_dir.parent, {"train": ["2025-06-01"], "val": ["2025-06-05"]})
    assert s == {"train": [seq_dir], "val": []}
    assert split_dirs(seq_dir.parent, {"train": ["2025-06-01"]}, ["Himawari-9"]) == {"train": []}
    with pytest.raises(AssertionError):
        split_dirs(seq_dir.parent, {"train": ["2025-06-01"], "val": ["2025-06-01"]})
