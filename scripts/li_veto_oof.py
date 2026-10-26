"""LiDAR veto of satellite kurgans (cross-modal precision filter), nested on training regions labelled in LiDAR AND
satellite: inside the LiDAR footprint, a satellite kurgan with no LiDAR kurgan within 6 m has its confidence × gamma
(gamma 0 = removed). LiDAR = OOF of the copy-paste/openness model (decoded as deployed), satellite = 4-member fusion
@0.25 (OOF), LiDAR → satellite twins α 0.1 as deployed. GT = all Li + SpOr labels of those regions.

  python scripts/li_veto_oof.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main():
    import geopandas as gpd
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from shapely.geometry import shape
    from koz3.fuse import fuse
    from koz3.labels import to_gt_geojson
    from koz3.li_train import labels, site_items, decode_site_v2
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from koz3.sp_fuse import fuse as sp_fuse
    from sp_assemble_oof import models
    from crossmodal_oof import df_to_feats
    items = {it["region"]: it for it in site_items(labels())}
    P = pickle.load(open(os.path.join(ROOT, "work/li/oof_augx.pkl"), "rb"))
    M = models(extra=["dfnodn"]); sp = sp_fuse(list(M.values()), np.ones(len(M))); sp = sp[sp.score >= 0.25]
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"].isin(["Li", "SpOr"])]
    both = set(L[L["mod"] == "Li"].region) & set(L[L["mod"] == "SpOr"].region) & set(P) & set(sp.region)
    regions = sorted(both); fold = {r: items[r]["fold"] for r in regions}
    print(len(regions), "regions with LiDAR + satellite labels and predictions")
    li, foot = {}, {}
    for r in regions:
        it = items[r]; meta = dict(it["meta"], region=r)
        li[r] = decode_site_v2(P[r].astype(np.float32), it["valid"], meta, thr=0.2, dual=0.05)
        for f in li[r]:
            f["properties"]["markup_type"] = "li"; f["properties"]["region_name"] = r
        ys, xs = np.where(it["valid"])
        tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True)
        X, Y = tr.transform(meta["x0"] + np.array([xs.min(), xs.max()]) * meta["res"], meta["ytop"] - np.array([ys.max(), ys.min()]) * meta["res"])
        foot[r] = (min(X), min(Y), max(X), max(Y), it)                   # bbox + valid mask (exact footprint test below)
    spf = {r: df_to_feats(sp[sp.region == r], "sp_or") for r in regions}
    folds = sorted(set(fold.values()))
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in regions if fold[r] == f])]) for f in folds}

    def in_foot(r, x, y):
        x0, y0, x1, y1, it = foot[r]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        meta = it["meta"]; tr = Transformer.from_crs(3857, meta["crs"], always_xy=True)
        X, Y = tr.transform(x, y); c, rr = int((X - meta["x0"]) / meta["res"]), int((meta["ytop"] - Y) / meta["res"])
        return 0 <= rr < it["valid"].shape[0] and 0 <= c < it["valid"].shape[1] and bool(it["valid"][rr, c])

    def make(s, f):
        gamma = s[0]; alpha = s[1] if len(s) > 1 else 0.1; out = []
        for r in regions:
            if fold[r] != f:
                continue
            lk = np.array([shape(g["geometry"]).centroid.coords[0] for g in li[r] if g["properties"]["class_name"] in K]).reshape(-1, 2)
            t = cKDTree(lk) if len(lk) else None
            sp_r = []
            for g in spf[r]:
                p = g["properties"]
                if gamma < 1 and p["class_name"] in K:
                    x, y = shape(g["geometry"]).centroid.coords[0]
                    k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
                    if in_foot(r, x, y) and (t is None or not t.query_ball_point((x, y), 6.0 * k)):
                        if gamma == 0:
                            continue
                        g = dict(g, properties=dict(p, confidence=p["confidence"] * gamma))
                sp_r.append(g)
            out += fuse(li[r] + sp_r, alpha={("li", "sp_or"): alpha} if alpha > 0 else {}, beta=0.0)
        return {"type": "FeatureCollection", "features": out}
    S = [(1.0, a) for a in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)] if os.environ.get("ALPHA") else [(1.0,), (0.5,), (0.3,), (0.1,), (0.0,)]
    C = fold_counts(make, S, folds, gt)
    for s in S:
        print("pooled", s, round(weighted(_add_all(C[s].values())), 4))
    h, ch, o, os_, tot = nested(C)
    print(f"first {weighted(_add_all(C[S[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {ch}")
    for c, v in tot.items():
        if v[0] + v[2]:
            print(f"   {c:24s} tp {v[0]} fp {v[1]} fn {v[2]}")


if __name__ == "__main__":
    main()
