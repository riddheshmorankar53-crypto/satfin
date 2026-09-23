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
