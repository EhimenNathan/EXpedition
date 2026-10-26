"""Diagnostic official-metric comparison on the fixed, geographic fold-0 sites.

Thresholds and weights below are predeclared; the best value on this fold is
diagnostic only and is not a deployment choice. A positive result needs other
held-out folds before packaging.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc


def main():
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "local_run_f0/oof_yolo_fold0.parquet"
    Y = pd.read_parquet(p)[["region", "cls", "score", "x", "y"]]
    M = models()
    RF = region_folds()
    # Include every held-out site, including sites on which YOLO emitted no
    # detections. Selecting sites from Y would condition the score on success.
    sites = {site for site, fold in RF.items() if fold == 0}
    assert sites
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(sites)]
    gt = to_gt_geojson(L)
    originals = [frame[frame.region.isin(sites)] for frame in M.values()]
    Y = Y[Y.region.isin(sites)]
    versions = {
        "yolo": Y,
        "reference": fuse(originals, np.ones(len(originals))),
        "reference+yolo@0.5": fuse(originals + [Y], np.array([1., 1., 1., .5])),
        "reference+yolo@1.0": fuse(originals + [Y], np.ones(len(originals) + 1)),
    }
    print("fold-0 sites", len(sites), "GT", len(L), "YOLO candidates", len(Y), flush=True)
    for name, P in versions.items():
        for t in (.20, .25, .30, .35, .40, .50):
            subset = P[P.score >= t]
            overall, details = score(fc(subset), gt)
            print(f"{name:20s} t={t:.2f} score={overall:.5f} n={len(subset)}", flush=True)


if __name__ == "__main__":
    main()
