"""Nested geographic OOF test of physical FCOS-box filters for Ae fortifications."""
import importlib.util
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = "/home/nathan/Expedition Competition"
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

spec = importlib.util.spec_from_file_location("ae_layer", os.path.join(ROOT, "Codex Work/ae_layer_oof.py"))
ae_layer = importlib.util.module_from_spec(spec); spec.loader.exec_module(ae_layer)

TH = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.1)
MAX_AREA = (1000, 2000, 5000, 10000, 20000, np.inf)
MIN_EDGE = (0, 10, 25, 50, 75)


def f1(v):
    tp, fp, fn = v
    return 2 * tp / max(2 * tp + fn + 0.5 * fp, 1e-12)


def main():
    D = pd.read_parquet(os.path.join(ROOT, "work/ae/oof_ae_fcos_fast_layers_boxmeta.parquet"))
    D = D[D.cls == "fortifikatsii"]
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    L = L[(L["mod"] == "Ae") & (L.class_name == "fortifikatsii") & L.region.isin(D.region.unique())]
    G = ae_layer.layer_gt(L)
    rf = D.groupby("region").fold.first().astype(int).to_dict(); folds = sorted(set(rf.values()))
    GT = {f: to_gt_geojson(G[G.region.map(rf) == f]) for f in folds}
    params = [(t, a, e) for t in TH for a in MAX_AREA for e in MIN_EDGE]
    C = {}
    for pi, (t, a, e) in enumerate(params):
        keep = D[(D.score >= t) & (D.box_area_m2 <= a) & (D.edge_m >= e)]
        for fold in folds:
            _, r = score(fc(keep[keep.region.map(rf) == fold]), GT[fold], classes=["fortifikatsii"])
            q = r["fortifikatsii"]
            C[(pi, fold)] = np.array((q["tp"], q["fp"], q["fn"]), int)

    base = {}
    for f in folds:
        # Deployed OOF rule: score >= .4, no physical box filter.
        p = params.index((0.4, np.inf, 0))
        base[f] = C[(p, f)]

    total = np.zeros(3, int); chosen = []
    for fold in folds:
        def train_metric(pi):
            return f1(sum((C[(pi, g)] for g in folds if g != fold), np.zeros(3, int)))
        pi = max(range(len(params)), key=train_metric)
        chosen.append(params[pi]); total += C[(pi, fold)]
        print("fold", fold, "choice", params[pi], "base", tuple(base[fold]), f1(base[fold]),
              "test", tuple(C[(pi, fold)]), f1(C[(pi, fold)]))
    btot = sum(base.values(), np.zeros(3, int))
    print("base pooled", tuple(btot), f1(btot))
    print("nested pooled", tuple(total), f1(total), "choices", chosen)
    pooled = {pi: sum((C[(pi, f)] for f in folds), np.zeros(3, int)) for pi in range(len(params))}
    pi = max(pooled, key=lambda q: f1(pooled[q]))
    print("oracle pooled", params[pi], tuple(pooled[pi]), f1(pooled[pi]))


if __name__ == "__main__":
    main()
