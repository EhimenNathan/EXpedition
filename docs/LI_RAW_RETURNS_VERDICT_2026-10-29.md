# Raw LiDAR return attributes: rejected

The deployed LiDAR models use the terrain surface and derived relief channels.
This experiment tested whether information retained in the original point cloud
but absent from the terrain image could improve candidate ranking:

- local point density;
- multiple-return share;
- intensity;
- core-versus-annulus contrasts at 2, 4, 8 and 16 metres;
- robust normalisation within each site to reduce vendor and sensor shift.

All 3,518 deployed candidates had raw-return coverage. Models were trained and
evaluated out of fold by geographic LiDAR fold.

| Evidence | Pooled held-out AUC |
|---|---:|
| CNN peak | 0.9166 |
| CNN confidence | 0.9057 |
| Existing morphometric GBM | **0.9263** |
| Raw returns only | 0.4127 |
| Morphometry + raw returns | 0.9243 |

Per-fold AUC, existing versus combined:

| Fold | Existing | + raw returns |
|---:|---:|---:|
| 0 | 0.9051 | 0.9046 |
| 1 | 0.9002 | 0.8901 |
| 2 | 0.9016 | 0.9063 |
| 3 | 0.9429 | 0.9452 |
| 4 | 0.9382 | 0.9401 |

Raw density, multiple returns and intensity add no stable discrimination at the
decision boundary. They should not be integrated or packaged. The result also
reduces concern that the deployed DTM discards a major vendor-specific signal.

Reproduction: `python experiments/codex/li_raw_returns_rescore_oof.py`.
