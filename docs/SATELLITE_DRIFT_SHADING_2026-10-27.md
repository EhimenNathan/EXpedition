# Satellite drift and illumination gates — 27 Oct 2026

Two previously pending CPU-only satellite tests were run against the deployed v20/v21 OOF pipeline. Neither passes the prospective acceptance gates.

## Public-composition retuning

Training has 20/129 large-scene satellite regions (15.5%), versus 17/30 (57%) on public. Importance weighting the OOF counts to the public mix produced:

- deployed weighted score: **0.3473**
- nested weighted score: **0.3538** (**+0.0065**)
- unweighted nested change: **+0.0012**
- outer-fold changes were mixed; folds 0 and 3 regressed, with fold 3 falling **0.2695 → 0.2428** under public-mix weights
- the selected large-scene threshold varied from 0.25 to 0.40 across folds

The gain is too small and unstable to deploy. The existing satellite settings remain unchanged.

## Illumination-consistency rescoring

Candidate scores were rescaled by the agreement between each candidate's brightness dipole and the leave-one-out scene illumination direction.

- deployed: **0.3797**
- nested: **0.3801** (**+0.0004**)
- oracle: **0.3816** at lambda 0.25
- worst outer-fold change: **-0.0027**
- fixed lambda 0.25 site test: **+0.0020**, k*=4, bootstrap P(gain>0)=0.82, 90% CI **[-0.0015, +0.0052]**

The confidence interval crosses zero and an outer fold regresses. Rejected.

Logs were written to `logs/sp_composition_oof_20261027.log` and `logs/sp_shading_oof_20261027.log` in the `feat/physics-gates` worktree.
