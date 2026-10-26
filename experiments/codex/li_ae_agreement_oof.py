#!/usr/bin/env python3
"""Prospective cross-modal confirmation for repeated Li/Ae observations.

For sites carrying both LiDAR and aerial film, duplicate a confident LiDAR
kurgan prediction into the aerial observation only when an independent aerial
OOF detector places a kurgan nearby.  All choices are evaluated by geographic
outer folds against the official pooled matcher.
"""

import copy
import glob
import json
import os
import sys
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import shape

ROOT = "/home/nathan/Expedition Competition"
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

K = ("kurgany_povrezhdennye", "kurgany_tselye")
TWIN_CLASSES = K + ("gorodishcha", "fortifikatsii")


def main():
    from koz3.li_train import labels, site_items, decode_site_v2
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    import li_compare_oof as lc

    all_labels = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    li_labels = labels()
    items = {it["region"]: it for it in site_items(li_labels)}
    probs = {nm: lc._pk(os.path.join(ROOT, f"work/li/oof_{nm}.pkl.zst")) for nm in ("ens", "aug", "augx")}
    ae_regions = set(all_labels.loc[all_labels["mod"] == "Ae", "region"])
    fold = {r: items[r]["fold"] for r in probs["ens"] if r in ae_regions}
    # Joint-model fold 1 has a known 18.951 km source/holdout separation where
    # the prospective protocol required 25 km.  Exclude it from strict evidence.
    sites = sorted(r for r in fold if fold[r] != 1)

    # Observable aerial coverage, independent of labels.
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import unary_union
    footprints = {}
    for r in sites:
        boxes = []
        for path in glob.glob(os.path.join(ROOT, "data/train-expds-3", r, "*_Ae_*", "*.tif")):
            with rasterio.open(path) as src:
                crs = src.crs.to_string() if src.crs is not None and src.crs.to_epsg() else json.load(open(os.path.join(ROOT, "data/train-expds-3", r, "UTM.json")))["crs"]
                tr = Transformer.from_crs(crs, 3857, always_xy=True); b = src.bounds
                xx, yy = tr.transform([b.left, b.right, b.right, b.left], [b.bottom, b.bottom, b.top, b.top])
                boxes.append(box(min(xx), min(yy), max(xx), max(yy)))
        footprints[r] = unary_union(boxes) if boxes else None
    ext = pd.read_parquet(os.path.join(ROOT, "work/ae/sheet_extent.parquet"))
    if ext.index.name != "region":
        ext = ext.set_index("region")

    prior = {"R": 200.0, "K": 3, "snb": 0.2}
    decoded = {}
    for r in sites:
        p = ((probs["ens"][r].astype(np.float32) + probs["aug"][r] + probs["augx"][r]) / 3).astype(np.float32)
        it = items[r]
        decoded[r] = decode_site_v2(p, it["valid"], dict(it["meta"], region=r),
                                    thr=0.15, dual=0.05, comp_thr=0.5, prior=prior)

    # These are genuine OOF predictions produced by models that excluded the
    # corresponding site's geographic fold.
    cand_paths = {
        "fcos": "work/ae/oof_ae_fcos_fast_cands.parquet",
        "joint": "work/ae/oof_ae_dfj_cands.parquet",
        "spdet": "work/ae/oof_spdet_cands.parquet",
    }
    ae = {}
    for name, rel in cand_paths.items():
        d = pd.read_parquet(os.path.join(ROOT, rel))
        ae[name] = d[d.region.isin(sites) & d.cls.isin(TWIN_CLASSES)].copy()

    # Build candidate trees lazily for each detector/threshold/site/family.
    trees = {}
    def nearby(det, athr, region, family, x, y, radius_m):
        key = (det, athr, region, family)
        if key not in trees:
            d = ae[det]
            h = d[(d.region == region) & d.cls.isin(K) & (d.score >= athr)]
            if family == "same":
                # strict class handled below using two trees
                raise RuntimeError("strict class needs a class key")
            arr = h[["x", "y"]].to_numpy(float)
            trees[key] = cKDTree(arr) if len(arr) else None
        tr = trees[key]
        if tr is None:
            return False
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
        return bool(tr.query_ball_point((x, y), radius_m * k))

    def nearby_cls(det, athr, region, cls, x, y, radius_m):
        key = (det, athr, region, cls)
        if key not in trees:
            d = ae[det]
            h = d[(d.region == region) & (d.cls == cls) & (d.score >= athr)]
            arr = h[["x", "y"]].to_numpy(float)
            trees[key] = cKDTree(arr) if len(arr) else None
        tr = trees[key]
        if tr is None:
            return False
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
        return bool(tr.query_ball_point((x, y), radius_m * k))

    def near_count(det, athr, region, x, y, radius_m):
        key = (det, athr, region, "kurgan")
        if key not in trees:
            d = ae[det]
            h = d[(d.region == region) & d.cls.isin(K) & (d.score >= athr)]
            arr = h[["x", "y"]].to_numpy(float)
            trees[key] = cKDTree(arr) if len(arr) else None
        tr = trees[key]
        if tr is None:
            return 0
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
        return len(tr.query_ball_point((x, y), radius_m * k))

    settings = [("base",)]
    settings += [("all_fp", t) for t in (0.25, 0.4)]
    settings += [("frame", 0.4, x) for x in (0.2, 0.3, 0.4, 0.5, 0.65, 0.8)]
    settings += [("frame_agree", 0.4, x) for x in (0.3, 0.5, 0.8)]
    settings += [("nbr", 0.4, rad, n) for rad in (50.0, 100.0, 200.0) for n in (2, 3, 5, 10)]
    for det in ("fcos", "joint"):
        for strict in (False, True):
            for athr in (0.2, 0.3, 0.4):
                for lthr in (0.15, 0.25, 0.4):
                    for rad in (4.0, 8.0, 12.0):
                        settings.append(("agree", det, strict, athr, lthr, rad))
    # Agreement by either complementary aerial detector.
    for athr in (0.2, 0.3, 0.4):
        for lthr in (0.15, 0.25, 0.4):
            settings.append(("either", False, athr, lthr, 8.0))
    for cap in (2, 3, 4):
        settings.append(("multi", "joint", 0.2, 0.4, 12.0, cap))
    settings.append(("agree_all", "joint", 0.2, 0.4, 12.0))

    def site_features(st, r):
        fs = list(decoded[r])
        if st[0] == "base":
            return fs
        mode = st[0]
        if mode == "all_fp":
            _, lthr = st
            fp = footprints[r]
            for f in decoded[r]:
                p = f["properties"]
                if p["class_name"] in K and p["confidence"] >= lthr and fp is not None and fp.contains(shape(f["geometry"]).centroid):
                    t = copy.deepcopy(f); t["properties"]["markup_type"] = "ae"
                    t["properties"]["confidence"] *= 0.1; fs.append(t)
            return fs
        if mode in ("frame", "frame_agree"):
            _, lthr, rho = st
            fp = footprints[r]
            if r not in ext.index:
                return fs
            e = ext.loc[r]
            for f in decoded[r]:
                p = f["properties"]
                if p["class_name"] not in K or p["confidence"] < lthr or fp is None:
                    continue
                q = shape(f["geometry"]).centroid
                if not fp.contains(q) or np.hypot(q.x-e.cx, q.y-e.cy) > rho*e.half:
                    continue
                if mode == "frame_agree" and not nearby("joint", 0.2, r, "kurgan", q.x, q.y, 12.0):
                    continue
                t = copy.deepcopy(f); t["properties"]["markup_type"] = "ae"
                t["properties"]["confidence"] *= 0.1; fs.append(t)
            return fs
        if mode == "nbr":
            _, lthr, rad, need = st
            fp = footprints[r]
            ks = [f for f in decoded[r] if f["properties"]["class_name"] in K and f["properties"]["confidence"] >= .15]
            xy = np.array([shape(f["geometry"]).centroid.coords[0] for f in ks])
            tree = cKDTree(xy) if len(xy) else None
            for f in decoded[r]:
                p = f["properties"]
                if p["class_name"] not in K or p["confidence"] < lthr or fp is None or tree is None:
                    continue
                q = shape(f["geometry"]).centroid
                if not fp.contains(q):
                    continue
                scale = 1 / np.cos(np.radians(np.degrees(2*np.arctan(np.exp(q.y/6378137.0))-np.pi/2)))
                if len(tree.query_ball_point((q.x, q.y), rad*scale)) < need:
                    continue
                t=copy.deepcopy(f);t["properties"]["markup_type"]="ae";t["properties"]["confidence"]*=.1;fs.append(t)
            return fs
        if mode == "agree_all":
            _, det, athr, lthr, rad = st
            for f in decoded[r]:
                p = f["properties"]
                if p["class_name"] not in TWIN_CLASSES or p["confidence"] < lthr:
                    continue
                q = shape(f["geometry"]).centroid
                if p["class_name"] in K:
                    ok = nearby(det, athr, r, "kurgan", q.x, q.y, rad)
                else:
                    ok = nearby_cls(det, athr, r, p["class_name"], q.x, q.y, rad)
                if ok:
                    t = copy.deepcopy(f); t["properties"]["markup_type"] = "ae"
                    t["properties"]["confidence"] *= 0.1; fs.append(t)
            return fs
        if mode == "multi":
            _, det, athr, lthr, rad, cap = st
            for f in decoded[r]:
                p = f["properties"]
                if p["class_name"] not in K or p["confidence"] < lthr:
                    continue
                q = shape(f["geometry"]).centroid
                for _ in range(min(cap, near_count(det, athr, r, q.x, q.y, rad))):
                    t = copy.deepcopy(f)
                    t["properties"]["markup_type"] = "ae"
                    t["properties"]["confidence"] *= 0.1
                    fs.append(t)
            return fs
        if mode == "agree":
            _, det, strict, athr, lthr, rad = st
            dets = (det,)
        else:
            _, strict, athr, lthr, rad = st
            dets = ("fcos", "joint")
        for f in decoded[r]:
            p = f["properties"]
            if p["class_name"] not in K or p["confidence"] < lthr:
                continue
            q = shape(f["geometry"]).centroid
            ok = any(nearby_cls(det, athr, r, p["class_name"], q.x, q.y, rad)
                     if strict else nearby(det, athr, r, "kurgan", q.x, q.y, rad)
                     for det in dets)
            if ok:
                t = copy.deepcopy(f)
                t["properties"]["markup_type"] = "ae"
                t["properties"]["confidence"] *= 0.1
                fs.append(t)
        return fs

    folds = sorted({fold[r] for r in sites})
    gt_frame = all_labels[all_labels.region.isin(sites) & all_labels["mod"].isin(["Li", "Ae"])]
    gt = {f: to_gt_geojson(gt_frame[gt_frame.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    counts = fold_counts(
        lambda st, f: {"type": "FeatureCollection", "features": [x for r in sites if fold[r] == f for x in site_features(st, r)]},
        settings, folds, gt)
    base = weighted(_add_all(counts[("base",)].values()))
    honest, chosen, oracle, oracle_st, _ = nested(counts)
    print(f"sites={len(sites)} folds={folds} base={base:.6f} nested={honest:.6f} oracle={oracle:.6f} setting={oracle_st}")
    print("chosen", list(chosen.values()))
    ranked = sorted(((weighted(_add_all(counts[s].values())), s) for s in settings), reverse=True)
    for value, st in ranked[:20]:
        print(f"pooled {value:.6f} delta {value-base:+.6f} {st}")

    from robust_accept import site_counts, report
    gt_site = {r: to_gt_geojson(gt_frame[gt_frame.region == r]) for r in sites}
    fixed = ("agree", "joint", False, 0.2, 0.4, 12.0)
    fixed_tests = [fixed, ("all_fp", 0.4), ("agree_all", "joint", 0.2, 0.4, 12.0)]
    fixed_tests += [("frame", 0.4, x) for x in (0.3, 0.5, 0.8)]
    fixed_tests += [("nbr", 0.4, 100.0, n) for n in (2, 3, 5)]
    for fixed in fixed_tests:
        sc = site_counts(lambda st, r: {"type": "FeatureCollection", "features": site_features(st, r)},
                         (("base",), fixed), sites, gt_site)
        report(f"Li/Ae repeated observations {fixed}", sc[("base",)], sc[fixed])
        for label, subset in (("small<=6km2", [r for r in sites if footprints[r] is not None and footprints[r].area <= 6e6]),
                              ("large>6km2", [r for r in sites if footprints[r] is not None and footprints[r].area > 6e6])):
            b = weighted(_add_all(sc[("base",)][r] for r in subset))
            t = weighted(_add_all(sc[fixed][r] for r in subset))
            print(f"  {label}: n={len(subset)} {b:.6f}->{t:.6f} delta={t-b:+.6f}")


if __name__ == "__main__":
    main()
