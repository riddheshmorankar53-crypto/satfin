#!/usr/bin/env bash
# Small default sample: 3 h of GOES-19 M1 band 13 (1-min cadence). Extra args are passed through.
set -e
python -m satfin.data.download --satellite goes19 --band 13 --start 2025-06-01T18:00 --hours 3 --sector M1 "$@"
