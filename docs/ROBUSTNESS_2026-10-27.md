# Private-set robustness audit — 27 October 2026

## 1. Value of one detection per class

Each class F1 is pooled over all regions: F1 = 2·TP / (GT + TP + 0.5·FP). Emitting a candidate pays only when its
calibrated probability of being a TP exceeds F1 / (4 − F1) (≈ 0.14 at F1 = 0.5). One extra TP is worth roughly
weight / (100 · GT) of final score. Public GT per class estimated from training labels per region × the public
region mix (30 satellite, 13 LiDAR, 4 aerial, 1 ortho, 2 MagGr, 1 EtGr, 1 TerFl):

| class | weight | est. public GT | score per extra TP |
|---|---:|---:|---:|
| hillforts | 15 | ~35 | **~0.0043** |
| architecture | 10 | ~90 | ~0.0011 |
| intact kurgans | 20 | ~300 | ~0.0007 |
| damaged kurgans | 30 | ~750 | ~0.0004 |
| fortifications | 5 | ~120 | ~0.0004 |
| geophysics objects | 10 | ~250 | ~0.0004 |
| finds | 10 | ~550 | ~0.0002 |

One hillfort TP is worth about ten kurgan TPs, and the v20 public gain (+0.0030) is below one hillfort. The same
small GT makes hillfort F1 the most volatile class between public and private sets.

## 2. Optical duplicates (branch fix/optical-one-object-per-data-type)

Training labels: 19 multi-image satellite regions, 129 labels in image overlaps, 0 per-image duplicate pairs; 3
multi-image aerial regions, no consistent-shift duplicates. The rules state that within one data type an object is
one polygon. The v20 per-image satellite/aerial emission therefore adds FPs in every multi-image region. The public
set has one such region (1EEH), which is why v20 was public-neutral; training has 14 % multi-image satellite regions.

## 3. Time budget and zero-score protection (branch fix/time-budget-watchdog)

v20 ran experts in sequence against one global deadline. `tests/test_time_budget.py` shows that a slow first expert
consumed the whole budget and the next expert ran 0/3 jobs; on the platform a slow LiDAR set would skip every
satellite region. A job that never returned (no deadline check inside `dtm.build`) would leave no result file.

Fix:
- each job's deadline reserves the prior cost of all later jobs, scaled by measured speed (never assumed faster
  than the slowest expert measured so far); with no pressure every expert gets the full slack, as before;
- LiDAR gains a no-TTA tier below 60 s of allowance;
- a watchdog writes the dedupe/fused partial result atomically at 28 min and exits; the write is guarded so the main
  thread and the watchdog can never both write.

Before packaging: fit the cost priors with `scripts/calibrate_budget.py` on a full rehearsal log and put the printed
`budget` block in `inference/models/config.json`. Accept only if the public-size rehearsal logs no
"time pressure" line and reproduces v20 features exactly; then re-run with `KOZ3_BUDGET_S=300` to watch the
degradation path and with `KOZ3_HARD_S=120` to watch the watchdog write.
