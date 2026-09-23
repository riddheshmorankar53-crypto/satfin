# SatFIN — Satellite Frame Interpolation Network

RIFE-style optical-flow frame interpolation to raise the temporal resolution of
geostationary satellite imagery. Trains on 1-min GOES mesoscale data, runs
inference on low-cadence data (10/30-min, incl. INSAT-3DS).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt       # CUDA: install torch from pytorch.org first
python -m satfin.env_check            # prints Python/PyTorch/GPU info
```

No GPU? Everything falls back to CPU with tiny settings.

## Data download

GOES ABI L1b radiances come from the public NOAA buckets (`noaa-goes16/18/19`, anonymous).
Mesoscale sectors (`M1`, `M2`) run at ~1-min cadence and give real ground truth.

```bash
python -m satfin.data.download --satellite goes19 --band 13 --start 2025-06-01T18:00 --hours 3 --sector M1
# or: bash scripts/download_goes.sh
```

Files land in `data/raw/<satellite>/<sector>/C<band>/`; existing files are skipped.
`--sector F|C` also works (full disk / CONUS, 10/5-min cadence). Defaults live in `configs/default.yaml`.
Mesoscale sectors can be moved between events, so check the sector center when picking dates.
