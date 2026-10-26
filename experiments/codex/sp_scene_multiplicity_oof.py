"""Exact OOF audit of satellite source-image multiplicity.

The organiser states that each optical source image is an independent reference
layer.  Existing inference pooled all images of a region before model fusion.
This script compares that behavior against fusion within each source image,
using the five deployed detection members and their original scene-aware OOF
outputs.  No detector is retrained and deployed thresholds are fixed.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkb as wkb
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform as geom_transform

ROOT = Path("/home/nathan/Expedition Competition")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_fusion_oof import fc, RU

RUNS = {
    "dfine": ROOT / "kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l",
    "rtdetr": ROOT / "kaggle_import/out/koz3-sp-rtdetr/sp_runs/sp_rtdetrv2_r50",
    "dfnodn": ROOT / "kaggle_import/code/work/sp_runs/sp_dfine_l_nodn8k",
    "balq": ROOT / "kaggle_import/code/work/sp_runs/sp_dfine_l_balq",
    "balq3": ROOT / "kaggle_import/code/work/sp_runs/sp_dfine_l_balq_s3",
}
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def crs_map():
    c = pd.read_csv(ROOT / "kaggle_import/code/docs/site_crs.csv").set_index("site")
    return {s: f"EPSG:{int(e)}" for s, e in c.json_epsg.items()}


def load_pred(path: Path, cmap: dict) -> pd.DataFrame:
    files = sorted(path.glob("oof_pred*.parquet"))
    # Prefer split 01/23 files; a directory can also contain a single 0123 file.
    split = [p for p in files if p.stem in ("oof_pred01", "oof_pred23")]
    if split: files = split
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    xs, ys = [], []
    for site, g in d.groupby("site"):
        tr = Transformer.from_crs(cmap[site], 3857, always_xy=True)
        cen = [wkb.loads(x).centroid for x in g.wkb]
        X, Y = tr.transform([p.x for p in cen], [p.y for p in cen])
        xs.append(pd.Series(X, index=g.index)); ys.append(pd.Series(Y, index=g.index))
    d["x"] = pd.concat(xs).sort_index(); d["y"] = pd.concat(ys).sort_index()
    d["cls"] = d.cls.map(lambda x: RU.get(x, x))
    return d.rename(columns={"site": "region"})[["region", "scene", "fold", "cls", "score", "x", "y"]]


def load_layer_gt(cmap: dict) -> gpd.GeoDataFrame:
    d = pd.read_parquet(RUNS["dfine"] / "oof_gt.parquet")
    rows = []
    for r in d.itertuples():
        tr = Transformer.from_crs(cmap[r.site], 3857, always_xy=True)
        g = geom_transform(tr.transform, wkb.loads(r.wkb))
        rows.append({"region": r.site, "scene": r.scene, "fold": int(r.fold),
                     "class_name": RU.get(r.cls, r.cls), "markup_type": "sp_or", "geometry": g})
    return gpd.GeoDataFrame(rows, crs=3857)


def fused(members: list[pd.DataFrame], per_scene: bool) -> pd.DataFrame:
    tagged = []
    for m in members:
        q = m.copy()
        q["key"] = q.region + ("||" + q.scene if per_scene else "")
        q = q.rename(columns={"region": "site"}); q["region"] = q.key
        tagged.append(q[["region", "cls", "score", "x", "y"]])
    z = fuse(tagged, np.ones(len(tagged)), match_m=8.0)
    if per_scene:
        q = z.region.str.split("||", regex=False, n=1, expand=True)
        z["region"], z["scene"] = q[0], q[1]
    else:
        z["scene"] = "pooled"
    # Fixed deployed thresholds and damaged/intact hedge.
    z = z[z.score >= np.where(z.cls == "arkhitektury", .15, .25)].copy()
    extra = []
    if "s_kp" in z:
        for r in z[z.cls.isin(K)].itertuples():
            other = K[1] if r.cls == K[0] else K[0]
            other_s = r.s_kt if other == K[1] else r.s_kp
            if other_s >= .4:
                d = r._asdict(); d["cls"] = other; d["score"] *= .99; extra.append(d)
    if extra:
        z = pd.concat([z, pd.DataFrame(extra)], ignore_index=True)
    return z


def stats(pred, gt, folds):
    out = {}; total = {}
    for f in sorted(set(folds.values())):
        p = pred[pred.region.map(folds) == f]
        g = gt[gt.fold == f]
        _, r = score(fc(p), to_gt_geojson(g), classes=list(K) + ["arkhitektury", "fortifikatsii", "gorodishcha"])
        out[f] = r
        for c, v in r.items():
            total[c] = np.add(total.get(c, (0, 0, 0)), (v["tp"], v["fp"], v["fn"]))
    return out, total


def pf(name, total):
    print(name)
    for c, v in total.items():
        tp, fp, fn = map(int, v)
        f = 2 * tp / max(2 * tp + fn + .5 * fp, 1)
        print(f"  {c:24s} {f:.6f} {tp}/{fp}/{fn}")


def main():
    cmap = crs_map()
    M = [load_pred(p, cmap) for p in RUNS.values()]
    regions = set.intersection(*(set(x.region) for x in M))
    M = [x[x.region.isin(regions)].copy() for x in M]
    fold_of = M[0].groupby("region").fold.first().astype(int).to_dict()
    gt = load_layer_gt(cmap); gt = gt[gt.region.isin(regions)].copy()
    print("regions", len(regions), "layer GT", len(gt), "models", [len(x) for x in M])
    pa = ROOT / "Codex Work/sp_region_pooled_five_oof.parquet"
    pb = ROOT / "Codex Work/sp_scene_preserved_five_oof.parquet"
    if pa.exists() and pb.exists():
        a, b = pd.read_parquet(pa), pd.read_parquet(pb)
    else:
        a = fused(M, False); b = fused(M, True)
        a.to_parquet(pa); b.to_parquet(pb)
    print("pred pooled", len(a), "per_scene", len(b))
    oa, ta = stats(a, gt, fold_of); ob, tb = stats(b, gt, fold_of)
    pf("pooled", ta); pf("per_scene", tb)
    print("per fold damaged pooled -> scene")
    for f in sorted(set(fold_of.values())):
        x, y = oa[f][K[0]], ob[f][K[0]]
        print(f, (x["tp"], x["fp"], x["fn"]), "->", (y["tp"], y["fp"], y["fn"]))


if __name__ == "__main__": main()
