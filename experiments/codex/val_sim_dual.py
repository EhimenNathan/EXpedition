"""Paired validation-shaped simulator for a fixed satellite dual-emission rule.

Reads the existing honest OOF expert outputs. Both arms use exactly the same
drawn regions and labels, so their paired difference isolates the new rule.
"""
import copy
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.score import score
from val_sim import COMPOSITION, build_predictions


def main():
    P = build_predictions()
    D = {}
    for region, features in P["SpOr"].items():
        extra = []
        for feature in features:
            prop = feature["properties"]
            if prop["class_name"] == "kurgany_tselye" and prop["confidence"] >= .35:
                new = copy.deepcopy(feature)
                new["properties"]["class_name"] = "kurgany_povrezhdennye"
                new["properties"]["confidence"] *= .05
                extra.append(new)
        D[region] = features + extra
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    regions = {m: sorted(set(L[L["mod"] == m].region)) for m in COMPOSITION}
    gt_region = {(m, r): L[(L["mod"] == m) & (L.region == r)] for m in COMPOSITION for r in regions[m]}
    rng = np.random.default_rng(20261023)
    rows = []
    for i in range(30):
        base, dual, gts = [], [], []
        for mod, n in COMPOSITION.items():
            pool = regions[mod]
            drawn = rng.choice(pool, min(n, len(pool)), replace=len(pool) < n)
            for region in drawn:
                region = str(region)
                pred = P.get(mod, {}).get(region, [])
                base.extend(pred)
                dual.extend(D.get(region, []) if mod == "SpOr" else pred)
                gts.append(gt_region[(mod, region)])
        gt = to_gt_geojson(pd.concat(gts))
        a = score({"type": "FeatureCollection", "features": base}, gt)[0]
        b = score({"type": "FeatureCollection", "features": dual}, gt)[0]
        rows.append((a, b, b - a))
        print(i, round(a, 4), round(b, 4), round(b - a, 5), flush=True)
    A = np.array(rows)
    result = {"draws": len(rows), "base_mean": A[:, 0].mean(), "dual_mean": A[:, 1].mean(),
              "delta_mean": A[:, 2].mean(), "delta_min": A[:, 2].min(),
              "positive_draws": int((A[:, 2] > 0).sum()), "rows": rows}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
