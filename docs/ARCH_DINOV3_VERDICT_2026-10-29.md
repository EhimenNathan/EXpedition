# DINOv3-SAT architecture probe verdict (2026-10-29)

## Verdict

Rejected for deployment. Frozen DINOv3-SAT tokens contain architecture signal, but the signal is not robustly additive to the deployed six-member satellite ensemble.

## Evidence

- Frozen-token LightGBM/logistic ensemble: token AUC **0.937**, AP **0.071**.
- LightGBM fusion: nested satellite score **0.3479 -> 0.3501**, but fold 0 regressed **-0.0063**; site bootstrap 90% CI **[-0.0006, +0.0190]** and sites up/down **4/4**.
- Logistic fusion: nested **0.3643 -> 0.3522**, with fold regressions **-0.0222** and **-0.0215**.
- Averaged LightGBM/logistic stack: nested **0.3739 -> 0.3660**; the oracle chose the deployed zero-weight setting and three folds regressed.

The candidate fails the prospective acceptance rules: no fold regression, k* >= 3, and bootstrap P(gain > 0) >= 0.8 with a positive confidence interval. No archive was built.

## Logs

- `logs/sp_arch_dino_ens_20261029.log`
- `logs/sp_archdino_fuse_20261029.log`
- `logs/sp_archdino_lr_fuse_20261029.log`
- `logs/sp_archdino_ens_fuse_20261029.log`
