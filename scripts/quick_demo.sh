#!/usr/bin/env bash
# Whole pipeline on a small sample: download, preprocess, short train, evaluate, infer.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m satfin.data.download --start 2025-06-01T18:00 --hours 3 --sector M1
python -m satfin.data.download --start 2025-06-05T18:00 --hours 2 --sector M1
python -m satfin.data.download --start 2025-06-10T18:00 --hours 2 --sector M2
python -m satfin.data.preprocess
[ -f runs/demo/best.pt ] || python -m satfin.train --run demo --steps "${STEPS:-500}"
python -m satfin.evaluate --ckpt runs/demo/best.pt --n "${N:-64}" --out outputs/demo/results.md
SEQ=$(ls -d data/processed/*20250610* | head -1)
python -m satfin.infer --ckpt runs/demo/best.pt --seq "$SEQ" --every 10 --k 9 --range 0:31 --out outputs/demo
echo "Done. Results in outputs/demo/. Dashboard: streamlit run app/dashboard.py"
