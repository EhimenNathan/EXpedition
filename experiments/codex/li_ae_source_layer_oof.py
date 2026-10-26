#!/usr/bin/env python3
"""Source-image-aware LiDAR to aerial repeated-observation OOF audit.

The organiser scores each aerial source image as an independent observation.
For every confident LiDAR kurgan, this audit emits at most one aerial twin per
source image, and only when the aerial OOF detector independently proposes a
nearby kurgan.  The emitted location can be the aerial proposal coordinate,
which avoids assuming perfect registration between LiDAR and historical film.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import shape

ROOT = Path("/home/nathan/Expedition Competition")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "Codex Work")]

from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.nested import fold_counts, nested, weighted, _add_all
from sp_fusion_oof import fc

K = ("kurgany_povrezhdennye", "kurgany_tselye")
ATHR = (0.15, 0.20, 0.30, 0.40)
LTHR = (0.15, 0.25, 0.40, 0.55)
RAD = (8.0, 12.0, 20.0, 30.0)


def main():
    from koz3.li_train import labels as li_train_labels, site_items, decode_site_v2
    import li_compare_oof as lc

    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    li_labels = li_train_labels()
    items = {it["region"]: it for it in site_items(li_labels)}
    probs = {nm: lc._pk(ROOT / f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    ae = pd.read_parquet(ROOT / "work/ae/oof_ae_fcos_fast_layers_cands.parquet")
    common = sorted(set(probs["ens"]) & set(ae.region) & set(labels[labels["mod"] == "Ae"].region))
    fold_of = {r: items[r]["fold"] for r in common}
    # The joint-model fold 1 has a known <25 km train/holdout separation.
    sites = [r for r in common if fold_of[r] != 1]
    folds = sorted({fold_of[r] for r in sites})
    ae = ae[ae.region.isin(sites) & ae.cls.isin(K)].copy()

    decoded = {}
    prior = {"R": 200.0, "K": 3, "snb": 0.2}
    for r in sites:
        p = ((probs["ens"][r].astype(np.float32) + probs["aug"][r] + probs["augx"][r]) / 3).astype(np.float32)
        it = items[r]
        decoded[r] = decode_site_v2(p, it["valid"], dict(it["meta"], region=r),
                                    thr=0.15, dual=0.05, comp_thr=0.5, prior=prior)

    # Reconstruct independent aerial source layers, while keeping LiDAR GT once.
    li_gt = labels[(labels["mod"] == "Li") & labels.region.isin(sites)]
    ae_gt0 = labels[(labels["mod"] == "Ae") & labels.region.isin(sites)]
    ae_gt = layer_gt(ae_gt0)
    gt_all = gpd.GeoDataFrame(pd.concat([li_gt, ae_gt], ignore_index=True), crs=labels.crs)
    gt = {f: to_gt_geojson(gt_all[gt_all.region.map(fold_of) == f]) for f in folds}

    cache = {}

    def scene_candidates(region, scene, athr, cls_key):
        key = region, scene, athr, cls_key
        if key not in cache:
            q = ae[(ae.region == region) & (ae.scene == scene) & (ae.score >= athr)]
            if cls_key in K:
                q = q[q.cls == cls_key]
            q = q.reset_index(drop=True)
            xy = q[["x", "y"]].to_numpy(float)
            cache[key] = (q, cKDTree(xy) if len(xy) else None)
        return cache[key]

    def site_features(st, region):
        fs = list(decoded[region])
        if st == ("base",):
            return fs
        athr, lthr, rad, class_mode, coord_mode = st
        scenes = ae.loc[ae.region == region, "scene"].unique()
        for feat in decoded[region]:
            prop = feat["properties"]
            if prop["class_name"] not in K or prop["confidence"] < lthr:
                continue
            q0 = shape(feat["geometry"]).centroid
            scale = 1 / np.cos(np.radians(np.degrees(2*np.arctan(np.exp(q0.y/6378137.0))-np.pi/2)))
            for scene in scenes:
                cls_key = prop["class_name"] if class_mode == "same" else "k"
                cand, tree = scene_candidates(region, scene, athr, cls_key)
                if tree is None:
                    continue
                ids = tree.query_ball_point((q0.x, q0.y), rad * scale)
                if not ids:
                    continue
                j = max(ids, key=lambda z: float(cand.iloc[z].score))
                row = cand.iloc[j]
                out_cls = prop["class_name"] if class_mode in ("same", "li") else str(row.cls)
                if coord_mode == "ae":
                    x, y = float(row.x), float(row.y)
                    # A tiny center polygon, matching deployed optical output.
                    geom = {"type": "Polygon", "coordinates": [[[x-1,y-1],[x+1,y-1],[x+1,y+1],[x-1,y+1],[x-1,y-1]]]}
                else:
                    geom = feat["geometry"]
                fs.append({"type": "Feature", "geometry": geom,
                           "properties": {"region_name": region, "markup_type": "ae",
                                          "class_name": out_cls,
                                          "confidence": min(float(prop["confidence"]), float(row.score))}})
        return fs

    settings = [("base",)] + [(a, l, r, cm, xy) for a in ATHR for l in LTHR for r in RAD
                               for cm in ("li", "same", "ae") for xy in ("ae", "li")]
    counts = fold_counts(
        lambda st, f: {"type": "FeatureCollection", "features":
                       [x for r in sites if fold_of[r] == f for x in site_features(st, r)]},
        settings, folds, gt)
    base = weighted(_add_all(counts[("base",)].values()))
    honest, chosen, oracle, oracle_st, _ = nested(counts)
    print(f"sites={len(sites)} folds={folds} GT Li={len(li_gt)} Ae={len(ae_gt0)} AeLayers={len(ae_gt)}")
    print(f"base={base:.6f} nested={honest:.6f} delta={honest-base:+.6f} oracle={oracle:.6f} setting={oracle_st}")
    print("chosen", list(chosen.values()))
    ranked = sorted(((weighted(_add_all(counts[s].values())), s) for s in settings), reverse=True)
    for value, st in ranked[:20]:
        print(f"pooled {value:.6f} delta {value-base:+.6f} {st}")

    # Site effects for the globally best fixed setting, useful for stability.
    best = ranked[0][1]
    from robust_accept import site_counts, report
    gt_site = {r: to_gt_geojson(gt_all[gt_all.region == r]) for r in sites}
    sc = site_counts(lambda st, r: {"type": "FeatureCollection", "features": site_features(st, r)},
                     (("base",), best), sites, gt_site)
    report(f"source-layer Li/Ae {best}", sc[("base",)], sc[best])


if __name__ == "__main__":
    main()
