"""Nested geographic OOF audit: unique high-confidence FCOS architecture proposals."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import _add_all, fold_counts, nested, weighted
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc


def unique_extra(base, candidates, dist_m=8.0):
    rows = []
    for region, g in candidates.groupby("region"):
        b = base[(base.region == region) & (base.cls == "arkhitektury")]
        g = g.sort_values("score", ascending=False)
        lat = np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)
        radius = dist_m / np.cos(np.radians(lat))
        btree = cKDTree(b[["x", "y"]].to_numpy()) if len(b) else None
        accepted = []
        for row in g.itertuples():
            point = np.array([row.x, row.y])
            if btree is not None and btree.query(point)[0] <= radius:
                continue
            if accepted and cKDTree(np.array(accepted)).query(point)[0] <= radius:
                continue
            accepted.append(point)
            rows.append((row.region, row.cls, row.score, row.x, row.y))
    return pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])


def main():
    M = models()
    regions = sorted(set.intersection(*(set(m.region) for m in M.values())))
    M = {k: v[v.region.isin(regions)] for k, v in M.items()}
    F = fuse(list(M.values()), np.ones(len(M)))
    F = F[F.score >= .25]
    X = pd.read_parquet(ROOT / "work/spor/oof_fcos_cands.parquet")
    X = X[(X.region.isin(regions)) & (X.cls == "arkhitektury") & (X.score >= .2)]
    X = unique_extra(F, X)
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    settings = [None, .2, .25, .3, .35, .4, .5, .6]

    def make(t, fold):
        b = F[F.region.map(RF) == fold]
        if t is None:
            return fc(b)
        a = X[(X.region.map(RF) == fold) & (X.score >= t)]
        return fc(pd.concat([b, a], ignore_index=True))

    C = fold_counts(make, settings, folds, GT)
    honest, chosen, oracle, oracle_t, _ = nested(C)
    result = {"regions": len(regions), "extra_candidates": len(X), "baseline": weighted(_add_all(C[None].values())),
              "nested": honest, "chosen_by_fold": {str(f): t for f, t in chosen.items()},
              "oracle": oracle, "oracle_threshold": oracle_t,
              "fixed": {str(t): {"pooled": weighted(_add_all(C[t].values())),
                                 "folds": {str(f): weighted(C[t][f]) for f in folds},
                                 "arch_counts": list(map(int, _add_all(C[t].values()).get("arkhitektury", (0,0,0))))}
                        for t in settings}}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
