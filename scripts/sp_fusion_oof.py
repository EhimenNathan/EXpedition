"""Satellite OOF fusion study: D-FINE-L, RT-DETRv2 (previous team, same folds) and our heatmap U-Net.

All candidates → EPSG:3857 points; GT = our SpOr labels of the OOF sites; official-exact scorer.
Fusion: per region and class family, candidates within MATCH_M are one object; its score is a weighted mean of the
models' scores (missing model = 0), class = the highest-scoring member's class.
"""
import os
import sys

import numpy as np
import pandas as pd
import shapely.wkb as wkb
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3.score import score  # noqa: E402
from koz3.sp_fuse import fuse  # noqa: E402,F401

RU = {"курганы_поврежденные": "kurgany_povrezhdennye", "курганы_целые": "kurgany_tselye", "городища": "gorodishcha",
      "архитектуры": "arkhitektury", "фортификации": "fortifikatsii"}
K = ("kurgany_povrezhdennye", "kurgany_tselye")
fam = lambda c: "k" if c in K else c


def load_theirs(d):
    P = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in ("oof_pred01.parquet", "oof_pred23.parquet")], ignore_index=True)
    return P


def to3857(P, crs_of):
    xs, ys = [], []
    for site, g in P.groupby("site"):
        tr = Transformer.from_crs(crs_of[site], 3857, always_xy=True)
        c = [wkb.loads(w).centroid for w in g.wkb]
        X, Y = tr.transform([p.x for p in c], [p.y for p in c])
        xs.append(pd.Series(X, g.index)); ys.append(pd.Series(Y, g.index))
    P = P.copy(); P["x"] = pd.concat(xs); P["y"] = pd.concat(ys)
    P["cls"] = P.cls.map(lambda c: RU.get(c, c))
    return P.rename(columns={"site": "region"})[["region", "cls", "score", "x", "y"]]


def fc(df, r=1.0):
    from shapely.geometry import Point
    feats = []
    for row in df.itertuples():
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(row.y / 6378137.0)) - np.pi / 2)))
        feats.append({"type": "Feature", "geometry": Point(row.x, row.y).buffer(r * k, 12).__geo_interface__,
                      "properties": {"class_name": row.cls, "confidence": float(row.score)}})
    return {"type": "FeatureCollection", "features": feats}


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    crs = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site")
    col = sys.argv[1] if len(sys.argv) > 1 else "auto_epsg"
    crs_of = {s: f"EPSG:{int(e)}" for s, e in crs[col].items()}
    D = to3857(load_theirs(os.path.join(ROOT, "kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l")), crs_of)
    R = to3857(load_theirs(os.path.join(ROOT, "kaggle_import/out/koz3-sp-rtdetr/sp_runs/sp_rtdetrv2_r50")), crs_of)
    H = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_hm_cands.parquet"))
    regions = sorted(set(D.region) & set(H.region))
    D, R, H = (m[m.region.isin(regions)] for m in (D, R, H))
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    gt = to_gt_geojson(L[(L["mod"] == "SpOr") & L.region.isin(regions)])
    print(len(regions), "regions; GT", len(gt["features"]))
    sweep = lambda name, df, ts: [print(f"{name:28s} t={t:.2f} {score(fc(df[df.score >= t]), gt)[0]:.4f}") for t in ts]
    sweep("D-FINE", D, (0.4, 0.5))
    sweep("RT-DETRv2", R, (0.4, 0.5))
    sweep("heatmap", H, (0.1, 0.15, 0.2))
    for w in ((1, 1, 0), (1, 1, 1), (1, 1, 0.5), (1, 0, 1), (2, 1, 1)):
        Fz = fuse([D, R, H], np.array(w, float))
        sweep(f"fuse w={w}", Fz, (0.2, 0.25, 0.3, 0.35, 0.4))


if __name__ == "__main__":
    main()
