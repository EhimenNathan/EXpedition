"""Validated test-time augmentation for the satellite D-FINE (previous team's fold models, our inference path).

Each fold model predicts ITS held-out scenes (their 0.5 m prep) under: base view; zoom ×Z (scene upsampled so small
mounds get more pixels, QASHI-style); and combinations (score-average or max over views, per object within 8 m).
Thresholds chosen nested over the 4 folds; official metric against our SpOr labels.
Candidates are cached to work/spor/tta_<variant>.parquet.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
PREV = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
SP = os.path.join(ROOT, "kaggle_import/code/work/sp")


def run_variant(zoom):
    out = os.path.join(ROOT, f"work/spor/tta_zoom{zoom}.parquet")
    if os.path.exists(out):
        return pd.read_parquet(out)
    from pyproj import Transformer
    from rasterio.transform import Affine
    from koz3.det_train import load_eager
    from koz3 import prev_sp_det
    S = pd.read_csv(os.path.join(SP, "scenes.csv")).set_index("scene")
    F = json.load(open(os.path.join(ROOT, "kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/folds.json")))
    rows = []
    for fold in range(4):
        m = load_eager("ustc-community/dfine-large-coco", os.path.join(ROOT, f"kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/fold{fold}.pt"))
        for sc in [s for s, f in F.items() if f == fold]:
            r = S.loc[sc]; img = np.load(os.path.join(SP, "img", sc + ".npy")); T = Affine(*json.loads(r["T"]))
            if zoom != 1.0:
                H, W = img.shape[1:]
                if H * W * zoom * zoom > 80e6:
                    continue      # keep runtime bounded (zoom is for small objects; huge scenes stay at base)
                img = np.stack([cv2.resize(ch, (int(W * zoom), int(H * zoom)), interpolation=cv2.INTER_LINEAR) for ch in img])
                T = T * Affine.scale(1 / zoom, 1 / zoom)
            tr = Transformer.from_crs(r["crs"], 3857, always_xy=True)
            for k, e, s in prev_sp_det.detect_scene([m], img, T, bs=4):
                X, Y = tr.transform(e.centroid.x, e.centroid.y)
                rows.append((r["site"], PREV[k], s, X, Y, fold))
        print(f"zoom {zoom}: fold {fold} done, {len(rows)} candidates", flush=True)
        del m
    df = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y", "fold"])
    df.to_parquet(out)
    return df


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    from koz3.sp_fuse import fuse
    from sp_fusion_oof import fc
    base = run_variant(1.0); z = run_variant(1.5)
    fold = dict(zip(base.region, base.fold)); fold.update(dict(zip(z.region, z.fold)))
    regions = sorted(fold); folds = sorted(set(fold.values()))
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(fold) == f]) for f in folds}
    zr = set(z.region)
    variants = {
        "base": base,
        "zoom": pd.concat([z, base[~base.region.isin(zr)]]),             # zoomed where run, base elsewhere
        "avg": fuse([base[["region", "cls", "score", "x", "y"]], z[["region", "cls", "score", "x", "y"]]], np.ones(2)),
        "max": fuse([base[["region", "cls", "score", "x", "y"]], z[["region", "cls", "score", "x", "y"]]], np.array([1.0, 0.0])),
    }
    # "max": score of the base model wherever it exists; use the true max over views instead
    both = pd.concat([base.assign(v=0), z.assign(v=1)], ignore_index=True)
    from scipy.spatial import cKDTree
    mx = []
    for (reg, c), g in both.assign(fam=both.cls.map(lambda q: "k" if q.startswith("kurgany") else q)).groupby(["region", "fam"]):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        t = cKDTree(g[["x", "y"]].values); used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            for j in t.query_ball_point(g.loc[i, ["x", "y"]].values, 8.0):
                used[j] = True
            mx.append((reg, g.cls[i], g.score[i], g.x[i], g.y[i]))
    variants["max"] = pd.DataFrame(mx, columns=["region", "cls", "score", "x", "y"])
    for name, D in variants.items():
        D = D.copy(); D["fold"] = D.region.map(fold)
        C = fold_counts(lambda t, f: fc(D[(D.score >= t) & (D.fold == f)]), [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6], folds, gt)
        h, ch, o, os_, _ = nested(C)
        print(f"{name:5s} honest {h:.4f}  oracle {o:.4f} @ {os_}  chosen {ch}")


if __name__ == "__main__":
    main()
