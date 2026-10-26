"""Geographic nested audit of existing satellite expert weights and threshold."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import fold_counts, nested
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc


def main():
    M = models()
    regions = sorted(set.intersection(*(set(v.region) for v in M.values())))
    M = {k: v[v.region.isin(regions)] for k, v in M.items()}
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    weights = {
        "equal": [1, 1, 1], "dfine2": [2, 1, 1], "rtdetr2": [1, 2, 1],
        "heatmap_half": [1, 1, .5], "heatmap2": [1, 1, 2], "dfine2_heatmap_half": [2, 1, .5],
    }
    ts = [round(float(t), 3) for t in np.arange(.15, .501, .025)]
    result = {"model_order": list(M), "regions": len(regions), "weights": {}}
    for name, weight in weights.items():
        F = fuse(list(M.values()), np.array(weight, float))

        def make(t, fold):
            return fc(F[(F.score >= t) & (F.region.map(RF) == fold)])

        C = fold_counts(make, ts, folds, GT)
        honest, chosen, oracle, oracle_t, by_class = nested(C)
        result["weights"][name] = {
            "nested": float(honest), "chosen": {str(k): v for k, v in chosen.items()},
            "oracle": float(oracle), "oracle_threshold": oracle_t,
            "class_counts": {k: list(map(int, v)) for k, v in by_class.items()},
        }
        print(name, result["weights"][name], flush=True)
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
