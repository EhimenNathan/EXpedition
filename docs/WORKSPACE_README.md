# KOZ-3 «Экспедиция DS. Комплексирование» — working repository

Goal: beat the public leaderboard leader (0.57645 on 22 Oct) and win the closed final (17–19 Nov 2026).

## Standing rule: check for organiser changes before every work session

The organisers replace files in place (often with the same size) and re-upload materials without notice.
Before every session, every training run and every upload:

```bash
python scripts/s3_changes.py          # bucket train-expds-3: NEW / MODIFIED / DELETED by size + ETag + mtime
python scripts/org_links_check.py     # every organiser Yandex.Disk link: md5 / size / modified date
```

Also check the platform task page (https://expds-platform.upgreat.one, needs login) for its "Изменения в датасете"
list, and the channels https://t.me/expds and https://t.me/expds_support. Record every change in `docs/CHANGES.md`
and re-prepare whatever it touches.

## Layout

| Path | What |
|---|---|
| `inference/` | the submission: `main.py` (platform entry), `koz3_infer/` (orchestrator, experts, atomic single write) |
| `koz3/` | shared training + inference code: labels, official-exact scorer, DTM, relief, heatmap heads, experts' trainers, cross-modal fusion |
| `scripts/` | data sync (S3, Yandex), streamed preps (LAS, Or), build, platform rehearsal |
| `compute_metrics.py` | official metric koz3-1.5.0 (md5 checked against the organisers' link) |
| `org/` | organiser materials (baseline, simple solution, docs, validation structure) |
| `kaggle_import/` | the previous team's code and trained models pulled from the private Kaggle account |
| `figures/` | every figure: organiser documents rendered page by page, data views, model diagnostics |
| `data/` | training data (git-ignored): labels, geophysics, satellite, Ae; LAS / Or are streamed into compact preps |
| `work/` | training runs, OOF predictions, rehearsals |

## Key facts that drive the design

* Metric: a hit needs the prediction's representative point inside the label and ≥ 50 % overlap of the smaller
  shape; point labels: centroid within 2 m. Matching ignores `markup_type`; a miss costs 2× a false alarm.
  Organisers (consultation 2): "contour accuracy does not affect the score".
* Zero-score rule: `result.geojson` must appear once, complete. We keep everything in memory and do one
  `os.replace` from `temp_result.geojson` at the end (`scripts/rehearse.py` verifies this with a 100 ms watcher).
* Platform image `wooferclaw/archeology:latest` (Python 3.12, torch 2.11, CUDA 12.8) has torch, rasterio,
  laspy, shapely, pyproj, scipy, cv2 — not geopandas / smp / timm. Models therefore ship as TorchScript and the
  inference path is geopandas-free.
* Kaggle / Colab with competition data is not automatically allowed (organiser answer, TR §13.11): train locally.

## Current submission (v2, 23 Oct): `build/koz3_submission_v2_20261023.zip` (1.36 GB)

Upload it on the platform as a qualifying attempt (zip with `metadata.json` in the root; image
`wooferclaw/archeology:latest` = the organisers' baseline image; entry `python inference/main.py`).

| Data type | Expert | Honest OOF |
|---|---|---:|
| LiDAR (LAS) | ConvNeXt-T U-Net on 9 relief channels; kurgan peaks + hillfort/fortification components | 0.522 |
| Satellite | D-FINE-L + RT-DETRv2 (previous team) + our heatmap U-Net, fused | 0.273 |
| Geophysics finds | physics candidates + LightGBM (numpy/GPU) | 0.719 |
| Geophysics objects | 0.5·CNN + 0.5·pixel-LightGBM | 0.353 |
| Ae | satellite D-FINE, damaged kurgans ≥ 0.7 | +0.002 pooled |
| Or | satellite D-FINE, architecture ≥ 0.5 | precision 0.13 |
| Cross-modal | LiDAR detections twinned into optical layers | 0.348 → 0.365 on multi-type regions |

Full record (including rejected ideas and why): `docs/EXPERIMENTS.md`. Figures: `figures/index.html`.

## Commands

```bash
python -m koz3.labels                         # labels → data/labels_3857.parquet
python -m koz3.score                          # self-test: fast scorer == official
python -m koz3.geo_prep && python -m koz3.geo_train --fold 0 ... --oof
python -m koz3.sp_prep --mod SpOr && python -m koz3.sp_train --mod SpOr --fold 0 ... --oof
python scripts/las_stream_prep.py && python -m koz3.li_train --cache --fold 0 ... --oof
python scripts/build_submission.py --geo ... --li ... --sp_or ... --config work/config.json
python scripts/rehearse.py build/koz3_submission.zip <val_copy>/input <val_copy>/gt.geojson
```

| v12 (final, 24 Oct) | build/koz3_submission_v12_final_20261024.zip | v11 + satellite framing prior, class hedge, 6 members, arch 0.15; MagGr objects 0.75; LiDAR d8 TTA | expected ≈ 0.46–0.47 |
