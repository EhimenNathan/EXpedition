# LiDAR half-scale inference audit — 27 October 2026

## Hypothesis

Erentuna has clean, high-SNR mounds but low OOF recall, and its mounds are physically larger than the dominant
training population.  A second inference pass on inputs downsampled 2x might move these mounds into the model's
learned pixel-size range without training new weights.

## Prospective test

`scripts/li_halfscale_oof.py` runs the existing seed-0 and seed-1 model for each held-out geographic fold on a 2x
downsampled relief tensor, resizes its seven probability maps back to the native grid, and blends them with the exact
deployed OOF blend.  The decoding threshold, dual-emission rule, component threshold and neighbourhood prior stay
fixed.  Blend alpha is selected on the other folds.

| half-scale alpha | pooled exact score | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (deployed) | 0.5884 | 0.6505 | 0.5148 | 0.5602 | 0.4251 | 0.5571 |
| 0.10 | 0.5863 | 0.6605 | 0.5098 | 0.5629 | 0.4116 | 0.5624 |
| 0.20 | **0.5911** | 0.6483 | 0.5060 | 0.5718 | 0.4252 | 0.5680 |
| 0.33 | 0.5743 | 0.6203 | 0.4882 | 0.5650 | 0.4353 | 0.5439 |
| 0.50 | 0.5023 | 0.4879 | 0.2282 | 0.4810 | 0.4188 | 0.4798 |

Nested selection chose alpha 0.20 on every outer fold, but its worst held-out fold change was **-0.0087**.  It fails
the untouched-fold non-regression gate.

The motivating site did not improve materially: Erentuna kurgan F1 0.3412 → 0.3417, with TP/FP/FN changing
18/37/51 → 17/27/52.  The pass removes false alarms but does not recover the large mounds.  Looted-ring sites also
regressed: Timerevo 0.5604 → 0.5348 and Zaolshanskaya 0.2532 → 0.1622.

## Decision

**Reject.** Do not package the half-scale pass or spend a public submission on SAHI/multi-scale inference from this
model.  The public 5,000–10,000 m² zero-recall bin is mostly satellite hillforts and giant ploughed damaged kurgans,
so it was not evidence that LiDAR tiling lacked receptive field.  Erentuna remains a representation/domain failure,
not a recoverable inference-scale mismatch.
