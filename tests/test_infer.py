import netCDF4
import numpy as np
import torch

from satfin.infer import interpolate, save_netcdf


class Blend(torch.nn.Module):
    """Stand-in model whose output is local (linear blend), so tiled and single-pass results must match."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, I0, I1, t):
        return {"pred": (1 - t.view(-1, 1, 1, 1)) * I0 + t.view(-1, 1, 1, 1) * I1}


def test_tiled_matches_single_pass():
    rng = np.random.default_rng(0)
    i0, i1 = rng.random((70, 90), np.float32), rng.random((70, 90), np.float32)
    ts = [0.25, 0.5, 0.75]
    full = interpolate(Blend(), i0, i1, ts, tile=1024)
    tiled = interpolate(Blend(), i0, i1, ts, tile=32, overlap=8, batch=2)
    for f, t_, t in zip(full, tiled, ts):
        np.testing.assert_allclose(t_, f, atol=1e-6)
        np.testing.assert_allclose(f, (1 - t) * i0 + t * i1, atol=1e-6)


def test_netcdf_roundtrip(tmp_path):
    frames = [np.full((4, 5), v, np.float32) for v in (0.0, 0.5, 1.0)]
    times = np.array(["2025-06-01T18:00:00", "2025-06-01T18:05:00", "2025-06-01T18:10:00"], "datetime64[s]")
    save_netcdf(tmp_path / "x.nc", frames, times, np.array([0, 1, 0], bool), (180.0, 330.0), {"band": 13, "c": [1, 2]})
    with netCDF4.Dataset(tmp_path / "x.nc") as ds:
        np.testing.assert_allclose(ds["bt"][:, 0, 0], [180, 255, 330])
        assert list(ds["interpolated"][:]) == [0, 1, 0]
        assert ds["time"][1] - ds["time"][0] == 300 and ds.source_band == 13
