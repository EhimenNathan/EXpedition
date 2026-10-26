"""Or (UAV orthophoto) expert.

Or-trained models failed out-of-site (heatmap: 0.000 with 20 km cluster folds, 0.005 leave-one-region-out — the
architecture labels sit in one 8 km site cluster). The satellite D-FINE (SpOr-trained) on the ortho resampled to
0.5 m finds architecture at 13 % precision (leave-one-region-out, 11 TP / 71 FP at conf 0.5) — above the ~5 % break-even
for the pooled architecture class — so only architecture ≥ 0.5 is emitted."""
from .ae import predict as _predict_det

FAMILIES = ["or"]


def READY():
    from .sp_or import _det_weights
    return bool(_det_weights("sp_dfine"))


def predict(region, paths, deadline):
    return _predict_det(region, paths, deadline, cfg_key="or", markup="or")
