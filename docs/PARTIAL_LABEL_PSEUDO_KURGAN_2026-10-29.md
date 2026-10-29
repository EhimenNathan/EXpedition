# Satellite partial-label pseudo-kurgan pilot (2026-10-29)

## Hypothesis

Satellite regions are annotated by site purpose. Regions without kurgan labels can still contain visible kurgans, so treating every such location as background may suppress recall. Six deployed OOF members were fused and high-confidence kurgan consensus candidates were added as pseudo labels only in regions with no kurgan annotations.

## Prospective setup

- Pseudo threshold: fused OOF score >= 0.45.
- Calibration audit on kurgan-annotated regions: 384 candidates, 77.6% within 3 m of a labelled kurgan.
- Added: 70 scene targets from 64 fused candidates in 26 regions (60 damaged, 10 intact).
- Detector: D-FINE-L no-denoising recipe, 8,000 iterations, fold 0 only.
- Geographic split: exact deployed fold map loaded from `sp_dfine_l/folds.json`.
- Evaluation: replace deployed `dfnodn` member inside the six-member satellite fusion; all other logic unchanged.

## Result

Rejected on the held-out fold.

| Setting | Score |
| --- | ---: |
| Base six-member fusion, threshold 0.25 | 0.3580 |
| Pseudo-label member replacement, threshold 0.25 | 0.3447 |
| Change | -0.0133 |

Site changes were 7 improved, 7 regressed and 23 unchanged; bootstrap P(gain > 0) = 0.12. The largest regressions were concentrated in the Kurgany satellite series. Training only on kurgan-annotated regions also failed separately, so neither exclusion nor high-confidence pseudo labels solved the partial-label problem.

Do not extend this pilot to the remaining folds.

## Class-specific negative-loss masking

A second prospective fold-0 pilot retained every scene and every annotation, but set the unmatched-query VFL loss to zero for the two kurgan channels in regions without any kurgan annotation. Other class losses, positive kurgan losses, box losses, training recipe and deployed geographic fold were unchanged.

At the fixed deployed fusion threshold 0.25, replacing `dfnodn` with the masked-loss member changed 0.3580 to 0.3552 (-0.0029). Sites improved/regressed/held: 8/9/20; bootstrap P(gain > 0) = 0.28 with 90% CI crossing zero. A threshold selected on the same single fold showed +0.0021 at 0.18, which is not an honest selection and does not justify more folds.

This variant is also rejected.
