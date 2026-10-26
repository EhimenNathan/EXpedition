# Continuation guide

## Repository scope

This repository preserves source code and decision evidence. It deliberately
does not contain competition imagery, labels, private Kaggle datasets, model
weights, OOF predictions, cached features, submission ZIPs, or Python
environments. Obtain those artifacts from the project owner and keep them in
the ignored paths shown below.

The historical workspace used this layout:

```text
data/                 downloaded and prepared competition data
work/                 checkpoints, OOF predictions, caches, rehearsals
kaggle_import/        imported legacy team models and OOF files
build/                staged and packaged submissions
koz3/                 shared training/scoring code
scripts/              orchestration, OOF, build and rehearsal scripts
inference/            platform entry point and expert inference code
experiments/codex/    additional Codex experiments and Kaggle kernels
```

Most scripts derive the project root from `__file__`, so clone the repository
at any path but restore artifacts beneath these relative directories.

## Environment

The platform runtime is `wooferclaw/archeology:latest` with Python 3.12,
PyTorch, CUDA, Rasterio, Shapely, PyProj, SciPy and OpenCV. Training used a
larger local environment described in `requirements-training.txt`. Install
optional research packages only for the experiment that imports them.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-training.txt
```

The organisers' S3 bucket uses public read-only credentials. Export them from
the competition page before running `scripts/s3_changes.py`:

```bash
export KOZ3_S3_ACCESS_KEY=...
export KOZ3_S3_SECRET_KEY=...
```

## Core workflow

```bash
# Prepare labels and verify the fast scorer against the official metric.
python -m koz3.labels
python -m koz3.score

# Geophysics.
python -m koz3.geo_prep
python -m koz3.geo_train --fold 0 --oof

# Satellite.
python -m koz3.sp_prep --mod SpOr
python -m koz3.sp_train --mod SpOr --fold 0 --oof

# LiDAR.
python scripts/las_stream_prep.py
python -m koz3.li_train --cache --fold 0 --oof

# Build and exact deployment rehearsal.
python scripts/build_submission.py --help
python scripts/rehearse.py build/submission.zip VALIDATION/input VALIDATION/gt.geojson
```

Training entry points for aerial/orthophoto detectors are in
`koz3/ae_det_train.py`. Foundation-model and candidate-verifier pilots are in
`experiments/codex/nvidia_pilot/` and
`experiments/codex/sp_candidate_verifier/`.

## Required acceptance gates

Do not promote a change from a pooled or randomly split score. Require:

1. Geographic or site-disjoint OOF using the official matched-object metric.
2. Nested selection, or a genuinely untouched final fold, for thresholds and
   postprocessing choices.
3. Non-regression across outer folds, with the fold-level counts retained.
4. Exact deployment parity using the packaged archive.
5. A full-system rehearsal that checks single atomic output write, runtime,
   object count, result size, and import availability.
6. Output limits: under 20,000 objects, 100 MB result, and 3,000 polygon
   vertices per object.

`compute_metrics.py` is the byte-verified official scorer. Its contour
diagnostics are auxiliary and do not affect `final_score`.

## Current deployment finding

The v19 EtGr/geophysics correction is documented in
`ETGR_LAYER_FIX_2026-10-26.md`. The paired 26-region rehearsal improved from
0.4635 to 0.4973 and reduced runtime from 310.4 to 250.7 seconds. The two
invariants are covered by `tests/test_etgr_layer_dedupe.py`:

- preserve coincident EtGr objects from separate source images;
- execute a mixed-family geophysics region once, with all its paths merged.

## Where to continue

Read `EXPERIMENTS.md` and `WEAKNESSES.md` before launching a run. They record
failed prospective tests that should not be repeated. The largest remaining
score losses are damaged/intact kurgans, aerial imagery, and fortifications.
New work should add genuinely different signal or repair a verified data/model
contract; threshold and member-weight searches are already saturated.

The Git history begins with the v19 deployment correction and then adds the
full continuation bundle. Preserve fold definitions and evaluator parity when
refactoring the original scripts.
