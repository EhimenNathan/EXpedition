"""Geographic OOF test of satellite-to-aerial confirmed twins.

The organiser defines repeated observations in separate source images as separate
reference objects.  A satellite detection is copied into an aerial layer only
when an independently trained aerial OOF detector has a nearby proposal in that
specific source image.  The emitted point is the aerial proposal coordinate,
which respects the documented cross-image georeferencing displacement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("/home/nathan/Expedition Competition")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "Codex Work")]

from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

K = ("kurgany_povrezhdennye", "kurgany_tselye")
S_THR = (0.20, 0.25, 0.35)
A_THR = (0.15, 0.20, 0.30)
RAD = (8.0, 12.0, 20.0, 30.0)


def f1(v):
    tp, fp, fn = v
    return 2 * tp / max(2 * tp + fn + 0.5 * fp, 1e-12)


def main():
    sat = pd.read_parquet(ROOT / "Codex Work/sp_region_pooled_five_oof.parquet")
    ae = pd.read_parquet(ROOT / "work/ae/oof_ae_fcos_fast_layers_cands.parquet")
    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    common = sorted(set(sat.region) & set(ae.region) & set(labels[labels["mod"] == "Ae"].region))
    sat = sat[sat.region.isin(common) & sat.cls.isin(K)].copy()
    ae = ae[ae.region.isin(common) & ae.cls.isin(K)].copy()
    fold_of = ae.groupby("region").fold.first().astype(int).to_dict()
    folds = sorted(set(fold_of.values()))

    # Satellite GT is already one row per reference.  Reconstruct aerial source
    # layers with the same footprint rule used to build the detector dataset.
    sp_gt = labels[(labels["mod"] == "SpOr") & labels.region.isin(common) & labels.class_name.isin(K)]
    ae_gt0 = labels[(labels["mod"] == "Ae") & labels.region.isin(common) & labels.class_name.isin(K)]
    ae_gt = layer_gt(ae_gt0)
    gt_all = gpd.GeoDataFrame(pd.concat([sp_gt, ae_gt], ignore_index=True), crs=labels.crs)
    gt = {f: to_gt_geojson(gt_all[gt_all.region.map(fold_of) == f]) for f in folds}

    # Trees are source-image specific; return one strongest proposal per scene.
    cache = {}
    def twins(sthr, athr, rad, cls_mode, coord_mode):
        base = sat[sat.score >= sthr][["region", "cls", "score", "x", "y"]].copy()
        add = []
        for r in base.itertuples(index=False):
            scenes = ae.loc[ae.region == r.region, "scene"].unique()
            scale = 1 / np.cos(np.radians(np.degrees(2*np.arctan(np.exp(r.y/6378137.0))-np.pi/2)))
            for scene in scenes:
                key = (r.region, scene, athr, r.cls if cls_mode == "same" else "k")
                if key not in cache:
                    q = ae[(ae.region == r.region) & (ae.scene == scene) & (ae.score >= athr)]
                    if cls_mode == "same": q = q[q.cls == r.cls]
                    arr = q[["x", "y"]].to_numpy(float)
                    cache[key] = (q.reset_index(drop=True), cKDTree(arr) if len(arr) else None)
                q, tree = cache[key]
                if tree is None: continue
                ids = tree.query_ball_point((r.x, r.y), rad * scale)
                if not ids: continue
                j = max(ids, key=lambda j_: float(q.iloc[j_].score))
                a = q.iloc[j]
                x, y = (float(a.x), float(a.y)) if coord_mode == "ae" else (float(r.x), float(r.y))
                cls = r.cls if cls_mode in ("same", "sat") else str(a.cls)
                add.append((r.region, cls, min(float(r.score), float(a.score)), x, y))
        if add:
            base = pd.concat([base, pd.DataFrame(add, columns=base.columns)], ignore_index=True)
        return base

    settings = [(s, a, r, cm, xy) for s in S_THR for a in A_THR for r in RAD
                for cm in ("sat", "same", "ae") for xy in ("ae", "sat")]
    counts = {}
    for n, st in enumerate(settings):
        pred = twins(*st)
        for f in folds:
            p = pred[pred.region.map(fold_of) == f]
            _, rr = score(fc(p), gt[f], classes=list(K))
            counts[(st, f)] = {c: np.array((rr[c]["tp"], rr[c]["fp"], rr[c]["fn"]), int) for c in K}
        if n % 100 == 0: print("setting", n, st, flush=True)

    def utility(d):
        # competition weights: damaged 30, intact 20
        return (30*f1(d[K[0]]) + 20*f1(d[K[1]])) / 50

    chosen, total = {}, {c: np.zeros(3, int) for c in K}
    for f in folds:
        def tr_score(st):
            d = {c: sum((counts[(st, g)][c] for g in folds if g != f), np.zeros(3, int)) for c in K}
            return utility(d)
        st = max(settings, key=tr_score); chosen[f] = st
        for c in K: total[c] += counts[(st, f)][c]

    pooled = []
    for st in settings:
        d = {c: sum((counts[(st, f)][c] for f in folds), np.zeros(3, int)) for c in K}
        pooled.append((utility(d), st, d))
    pooled.sort(reverse=True, key=lambda x: x[0])
    print("common", common)
    print("folds", fold_of, "GT", len(sp_gt), len(ae_gt0), len(ae_gt))
    print("nested", utility(total), "chosen", chosen)
    for c in K: print(c, f1(total[c]), tuple(map(int, total[c])))
    print("oracle top")
    for u, st, d in pooled[:15]: print(f"{u:.6f}", st, {c: (round(f1(d[c]),6), tuple(map(int,d[c]))) for c in K})


if __name__ == "__main__":
    main()
