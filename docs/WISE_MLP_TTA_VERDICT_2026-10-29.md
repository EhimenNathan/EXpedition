# WiSE-FT, label-free MLP gating and detector TTA verdict

## WiSE-FT

Interpolating the D-FINE no-denoising member toward its pretrained weights at
alpha 0.2 improved the seven-member satellite fusion from 0.3664 to 0.3707 in
the original geographic OOF evaluation. Three folds improved by 0.0154 to
0.0306, but fold 0 regressed by 0.0265. Robustness was insufficient: k*=2 and
the 90% site-bootstrap interval crossed zero.

## Label-free lightweight gate

A nested gate tested whether the unstable WiSE behavior could be predicted on
an unseen region from 59 label-free signals:

- candidate counts and score quantiles by class;
- base-versus-WiSE candidate agreement, displacement and score shift;
- scene-presence score;
- scene count, physical extent and small-scene share.

Inside every outer geographic fold, the model family and gate threshold were
selected from inner leave-one-fold-out predictions. Candidate models were
ridge regressors, a Huber regressor, depth-1/2 random forests and an 8-unit MLP
with strong regularisation.

The nested gated gain was only +0.0001. Fold changes were +0.0041, -0.0052,
+0.0005 and +0.0018. Robustness was k*=1, bootstrap P(gain>0)=0.58, with 90%
CI [-0.0034, +0.0042]. The MLP and the complete gate are rejected.

## Eight-view detector TTA

Eight dihedral views for the no-denoising D-FINE member improved its satellite
OOF result, but regressed a geographic fold. Packaged on top of the v26 DINOv3
hillfort stack, the 26-region paired rehearsal completed in 11:29 and scored
0.52714 versus the v26 control 0.5306. It emitted 3,235 versus 3,201 features.
The candidate is rejected and its archive was removed.

## Conclusion

Weight interpolation, a small MLP and more inference views can each alter the
errors, but none supplies a stable new decision boundary. Future work must add
independent spatial or physical evidence rather than further optimise the same
correlated detector scores.
