"""Exact LiDAR OOF test of the DINOv3-SAT relief localizer.

The deployed LiDAR blend is decoded unchanged.  DINO may replace the strongest
existing hillfort point, rescue a site with no decoded hillfort, or do both.
Settings are selected on the other geographic folds and reported per site.
"""
from __future__ import annotations

import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def circle(x, y):
    k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
    return Point(float(x), float(y)).buffer(float(k), 12).__geo_interface__


def main(points="work/hf_dinov3/li_loc_points_oof.csv"):
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from robust_accept import site_counts, report

    with open(ROOT.parent.parent / "work/li/oof_blend3.pkl", "rb") as fh:
        probs = pickle.load(fh)
    L = labels()
    items = {q["region"]: q for q in site_items(L) if q["region"] in probs}
    dino = pd.read_csv(ROOT / points).set_index("region")
    sites = sorted(items)
    folds = sorted({items[r]["fold"] for r in sites})
    base = {}
    prior = {"R": 200.0, "K": 3, "snb": 0.2}
    for i, r in enumerate(sites, 1):
        it = items[r]
        base[r] = decode_site_v2(probs[r].astype(np.float32), it["valid"], dict(it["meta"], region=r),
                                 thr=.15, dual=.05, comp_thr=.5, prior=prior)
        if i % 10 == 0:
            print("decoded", i, "/", len(sites), flush=True)
    del probs

    def change(r, setting):
        mode, gate = setting
        feats = copy.deepcopy(base[r])
        if mode == "base":
            return feats
        row = dino.loc[r]
        score = (float(row.map_max) + float(row.p_scene)) / 2
        if score < gate:
            return feats
        idx = [i for i, f in enumerate(feats) if f["properties"]["class_name"] == "gorodishcha"]
        if idx and mode in ("replace", "both"):
            best = max(idx, key=lambda i: feats[i]["properties"].get("confidence", 0))
            feats[best]["geometry"] = circle(row.x3857, row.y3857)
            feats[best]["properties"]["confidence"] = max(float(feats[best]["properties"].get("confidence", 0)), score)
        elif not idx and mode in ("rescue", "both"):
            feats.append({"type": "Feature", "geometry": circle(row.x3857, row.y3857),
                          "properties": {"class_name": "gorodishcha", "confidence": score,
                                         "region_name": r, "markup_type": "li"}})
        return feats

    settings = [("base", 0.)] + [(m, g) for m in ("replace", "rescue", "both") for g in (.4, .5, .6, .7, .8)]
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if items[r]["fold"] == f])]) for f in folds}
    counts = fold_counts(lambda st, f: {"type": "FeatureCollection", "features": sum(
        (change(r, st) for r in sites if items[r]["fold"] == f), [])}, settings, folds, gt, CLS)
    for st in settings:
        tot = _add_all(counts[st].values())
        gor = tuple(int(v) for v in tot.get("gorodishcha", (0, 0, 0)))
        print(st, "pooled", round(weighted(tot), 5), "fold",
              [round(weighted(counts[st][f]), 5) for f in folds], "gor", gor)
    honest, chosen, oracle, oracle_st, _ = nested(counts)
    base_score = weighted(_add_all(counts[("base", 0.)].values()))
    print("base", round(base_score, 5), "nested", round(honest, 5), "delta", round(honest-base_score, 5),
          "oracle", round(oracle, 5), oracle_st, "chosen", chosen)
    fixed = max(settings, key=lambda s: weighted(_add_all(counts[s].values())))
    gt_site = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    site = site_counts(lambda st, r: {"type": "FeatureCollection", "features": change(r, st)},
                       (("base", 0.), fixed), sites, gt_site)
    report(f"LiDAR DINO hillfort {fixed}", site[("base", 0.)], site[fixed])


if __name__ == "__main__":
    main(*sys.argv[1:])
