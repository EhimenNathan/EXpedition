# Aerial fortification recovery

## Finding

The v19 deployment disabled the complete aerial expert after aerial kurgan
predictions failed geographic validation.  That switch also disabled a separate
FCOS detector trained for `fortifikatsii`.  The two paths have different
targets and different validation behaviour, so they must be controlled
independently.

The v20 configuration enables the aerial expert, leaves `ae.classes` empty so
that no aerial kurgans are emitted, and retains only the independently trained
fortification detector at its pre-existing fixed threshold of 0.40.

This is consistent with the organiser clarification that a Second World War
fortification and a damaged mound may overlap and should be labelled as two
objects.  The final deduplicator therefore keeps different classes separate.
Predictions from different aerial source images also remain separate, because
each image has its own reference objects.

## Layer-aware geographic OOF

The audit expands aerial ground truth once per source-image footprint and keeps
candidate identity by source image.  There are 2,179 original aerial labels and
2,445 layer-aware labels across 32 regions and four fixed geographic folds.

At the fixed 0.40 threshold, fortification OOF counts are:

| Fold | TP | FP | FN | F1 |
|---:|---:|---:|---:|---:|
| 0 | 53 | 136 | 103 | 0.3827 |
| 1 | 21 | 325 | 102 | 0.1370 |
| 2 | 0 | 39 | 4 | 0.0000 |
| 3 | 40 | 252 | 85 | 0.2749 |
| **Pooled** | **114** | **752** | **294** | **0.253898** |

Fold 2 is the strict non-regression case.  The deployed satellite system emits
one false fortification there and finds none of the five source-layer targets
(four aerial plus one satellite), so its full-system fortification F1 is
already zero.  Adding 39 aerial false positives leaves that class F1 and the
weighted final score exactly unchanged on the fold.  The other geographic
folds recover true positives.

A region-level count/maximum-score gate was also tested.  Nested selection was
unstable and failed to suppress the fold-2 false site, so the gate is rejected.
The deployment uses the fixed model threshold without a data-mined site rule.

## Exact deployment rehearsal

The authoritative rehearsal ground truth preserves both satellite and aerial
source-image layers.  On the same 26-region full-system input:

| System | Final score | Fortification F1 | Fortification TP/FP/FN |
|---|---:|---:|---:|
| v19b | 0.481521 | 0.021739 | 2 / 8 / 176 |
| v20, aerial fort only | 0.5130 | 0.651 | 136 / 207 / 42 |
| v20, aerial fort + satellite layer fix | **0.5276** | **0.651** | **136 / 207 / 42** |

Other class scores are unchanged by the aerial fortification switch.  The
combined archive emitted 3,065 features, completed in 270 seconds, and wrote
the result atomically.  This is a strong local rehearsal gain, but the public
set composition is unknown.  A defensible public expectation is approximately
0.508-0.520 from the v19b score of 0.4988988962.  It is not evidence for a 0.60
public score by itself.

## Reproduction

- `scripts/ae_layer_oof.py`: layer-aware aerial OOF and nested threshold audit.
- `scripts/ae_fort_region_gate_oof.py`: rejected observable region-gate audit.
- `configs/deployed_v20.json`: exact configuration packaged with v20.
- `docs/OPTICAL_LAYER_AUDIT_2026-10-27.md`: source-image contract for optical
  modalities.

## Public result

The platform scored v20 at **0.5018959557**, versus v19b at **0.4988988962**:
a gain of **+0.0029970595**.  All non-aerial modality precision and recall
figures were unchanged.  Aerial precision was 0.102941 and recall 0.093333,
which uniquely reconstruct to 7 TP, 122 FP and 68 FN.  The satellite
source-layer correction was neutral on this public set.

This result validates the recovered detector but also shows that the local
26-region rehearsal over-represented aerial fortification benefit.  The public
score is now the best verified score; do not resubmit v20.

The tested archive is intentionally not stored in Git because it contains
model weights.  Its SHA-256 is
`9a3afaf1c2ef4979629e033b1732e4e566b343da67dab0c8e4a096e69e3b19c5`.
