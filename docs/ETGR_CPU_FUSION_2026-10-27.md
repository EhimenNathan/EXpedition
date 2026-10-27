# EtGr CPU probability fusion — 27 Oct 2026

## Platform control

`koz3_submission_v21_robust_watchdog_20261027.zip` scored **0.5018959557** in **12:54** on the platform, exactly matching v20 in every reported modality metric. This validates the scheduler/watchdog build without a score change.

## LiDAR dome rescue

Claude's v3 physical dome ranker was evaluated against the deployed LiDAR OOF ensemble with `scripts/li_dome_fusion_oof.py` on all 59 cached sites.

- deployed: **0.5884**
- nested: **0.5884**
- oracle: **0.5884**, setting `None`
- every outer fold selected `None`
- site robustness: 0 up / 0 down / 59 unchanged, k*=0

The standalone physical signal is real but adds no net value after the CNN. Rejected.

## EtGr-only comparison

Exact deployed EtGr counts at the current 50/50 CNN+GBM, threshold 0.30, 0.5 m² minimum, pole anchor:

| Region | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| 212 | 41 | 14 | 42 | 0.6260 |
| 213 | 4 | 3 | 12 | 0.3721 |
| 214 | 50 | 23 | 126 | 0.4211 |
| 215 | 75 | 113 | 34 | 0.6237 |
| 216 | 51 | 91 | 77 | 0.4543 |
| 217 | 49 | 101 | 96 | 0.4008 |
| **pooled** | **270** | **345** | **387** | **0.4911** |

A local rerun of `experiments/claude/geo_objects_cpu.py` gave nested EtGr F1 **0.5346** (348 TP, 594 FP, 309 FN), lower than the earlier reported 0.567 but still above deployed. Alone it regressed regions 212, 215 and 216.

Probability-level fusion was then evaluated per source layer, selecting on the other five regions. The single fixed no-regression setting was:

- 20% CPU probability + 80% deployed probability
- threshold 0.30
- minimum component area 0.05 m²
- fixed 0.25 m interior patch

This scored **0.5592** versus deployed **0.4911**. All six regions improved; the smallest gain was +0.0093. Nested selection scored 0.5513, while the best pooled setting scored 0.5768 but regressed two regions.

## Deployment parity and full-system gate

- all-data LightGBM: 1,549,007 sampled pixels, 55 features
- JSON model matches LightGBM exactly under `NumpyGBM` (max probability error 0)
- NumPy CIE Lab features match the training implementation: max feature error 4.5e-6, 99th percentile 2.8e-7
- archive integrity and single atomic write passed

Paired 26-region rehearsals against the same current input and GT:

| Build | Runtime | Features | object_poly | final_score |
|---|---:|---:|---:|---:|
| v21 control | 278.7 s | 3,065 | 0.506 | **0.5291** |
| EtGr fusion | 322.9 s | 3,201 | 0.521 | **0.5306** |

All unrelated class counts and scores were exactly identical in the paired run. The full-system gain is only **+0.0015**, so this fails the substantial-gain submission gate despite the clean EtGr signal. The candidate archive was deleted. The implementation remains asset-gated: without `inference/models/geo/etgr_cpu.json`, behavior is unchanged.
