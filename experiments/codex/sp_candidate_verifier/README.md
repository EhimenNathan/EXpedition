# Satellite candidate verification pilot

The present v14 satellite stack emits many visually plausible mound candidates in regions annotated primarily for hillforts. In the four local geographic OOF folds, kurgan candidates at score >=0.08 have an inside-label rate of 13.3% on folds 0-2 (681/5110) and 2.1% on fold 3 (115/5399). Fold 3 contains 35 sites with no kurgan label and 3465 candidates from those sites. This is an annotation and region-mixture shift, not just a confidence-calibration change.

`prepare.py` extracts 96x96 RGB crops at 0.5 m/pixel. Private Kaggle input contains 5081 training crops from folds 0-2 after excluding two sites within 25 true geodesic km of fold 3. Fold-3 input contains 5399 crops and row IDs, with **no heldout labels**. Pretrained ConvNeXt-T and SatlasPretrain aerial Swin-v2-B pilots train separately without heldout validation or checkpoint choice.

`score_fixed.py` was written before predictions were downloaded. It preserves the deployed stack and tests a high-confidence rescue of low-score candidates (p_obj >= .98), a stringent removal of baseline candidates (p_obj < .005), and their combination. These fixed cutoffs are diagnostic and require prospective geographic replication before deployment.

The existing hillfort scene classifier gives another conservative site-intent rule: if its probability is >= .9 and the strongest fused kurgan score is < .5, suppress kurgan outputs for that site. It improved exact satellite OOF from .36637 to .37109, with no kurgan-labeled site removed in the four folds. This is an exploratory component gain; full-system/public gain is unknown. The rule was examined after fold-3 site diagnostics, so fold 3 is not a pristine final holdout for this rule.

Model weights from any promising kernel must be checked for package/runtime parity before being considered for a submission. The existing satellite OOF base-model folds have four cross-fold site pairs closer than 25 true km, so a final strict nested validation would need regenerated buffered base predictions.
