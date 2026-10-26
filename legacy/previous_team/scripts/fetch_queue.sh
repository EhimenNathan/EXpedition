#!/bin/bash
# Download queue in priority order (resumable; each step skips finished files). Run from the project root.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
python scripts/s3_fetch.py data_geo "^2(0\d|1\d)_" --exclude "\.las$|_Li_|_MagGr_цвет\.tif$|^200_.*_MagGr_серый" > logs/fetch_geo.log 2>&1
python scripts/s3_fetch.py data_sp "_SpOr_.*\.tif$|_SpOr_.*\.geojson$|/UTM\.json$" --exclude "^2(0\d|1\d)_" > logs/fetch_sp.log 2>&1
python scripts/s3_fetch.py data_geo "^200_.*_MagGr_серый" > logs/fetch_geo200.log 2>&1
