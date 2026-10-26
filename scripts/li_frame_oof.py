"""LiDAR label framing (Vesuvius "supervision mask" view: labels are only trustworthy near the site), CPU, honest.

1) Diagnostic: per detection of the deployed decoder, d_rel = distance from the centre of the site's valid-data bbox /
   half of its longer side; TP rate by d_rel bin and by score (are far detections mostly FPs = unlabelled terrain?).
2) LiDAR framing prior, nested + robust: detections with d_rel > x need confidence ≥ t2 (as the satellite prior).
  python scripts/li_frame_oof.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    from pyproj import Transformer
    from shapely.geometry import shape
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from koz3.score import CALC
    from robust_accept import site_counts, report
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    sites = sorted(P["ens"]); fold = {r: items[r]["fold"] for r in sites}; folds = sorted(set(fold.values()))
    pri = {"R": 200.0, "K": 3, "snb": 0.2}; D, rows = {}, []
    for r in sites:
        it = items[r]; meta = it["meta"]
        b = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32)
        fs = decode_site_v2(b, it["valid"], dict(meta, region=r), thr=0.15, dual=0.05, comp_thr=0.5, prior=pri)
        v = np.asarray(it["valid"], bool) if it["valid"] is not None else np.ones(b.shape[1:], bool)
        ys, xs = np.nonzero(v); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2
        tr = Transformer.from_crs(3857, meta["crs"], always_xy=True)
        g = L[L.region == r]; gl = list(g.to_crs(meta["crs"]).geometry)
        for f in fs:
            c = shape(f["geometry"]).centroid; X, Y = tr.transform(c.x, c.y)
            px, py = (X - meta["x0"]) / meta["res"], (meta["ytop"] - Y) / meta["res"]
            f["_d"] = float(np.hypot(px - cx, py - cy) / max(half, 1))
            from shapely.geometry import Point
            near = any(q.distance(Point(X, Y)) < 2.0 for q in gl)
            rows.append((r, f["properties"]["class_name"], f["properties"]["confidence"], f["_d"], near))
        D[r] = fs
    R = pd.DataFrame(rows, columns=["region", "cls", "conf", "d", "near"])
    R["dbin"] = pd.cut(R.d, [0, 0.25, 0.5, 0.75, 1.0, 1.5])
    print("near-a-label rate (≈ TP rate) by d_rel bin:\n", R.groupby("dbin", observed=True).agg(n=("near", "size"), near=("near", "mean"), conf=("conf", "median")).round(3).to_string())
    R["cbin"] = pd.cut(R.conf, [0, 0.2, 0.3, 0.5, 1.0])
    print(R.pivot_table(index="dbin", columns="cbin", values="near", aggfunc="mean", observed=True).round(2).to_string(), flush=True)
    clean = lambda fs: {"type": "FeatureCollection", "features": [{k: v for k, v in f.items() if k != "_d"} for f in fs]}

    def site_fc(st, r):
        if st[0] == "base":
            return clean(D[r])
        x, t2 = st
        return clean([f for f in D[r] if f["_d"] <= x or f["properties"]["confidence"] >= t2])
    SS = [("base",)] + list(itertools.product((0.4, 0.55, 0.7, 0.85), (0.2, 0.25, 0.35, 0.5)))
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    C = fold_counts(lambda st, f: {"type": "FeatureCollection", "features": [g for r in sites if fold[r] == f for g in site_fc(st, r)["features"]]}, SS, folds, gt, CLS)
    for st in SS:
        print(f"{st}: pooled {weighted(_add_all(C[st].values())):.4f}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    CS = site_counts(site_fc, (SS[0], os_), sites, gtr, CLS)
    report(f"LiDAR framing prior {os_}", CS[SS[0]], CS[os_])


if __name__ == "__main__":
    main()
