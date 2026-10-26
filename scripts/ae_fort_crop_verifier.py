"""Geographically cross-fitted visual verifier for Ae fortification proposals.

FCOS remains the proposal generator.  Frozen DINOv2 crop features answer the
narrower question "does this proposed centre look like a labelled fortification?"
All verifier predictions are out of fold; operating points are selected on the
other geographic folds with the exact competition metric.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

ROOT = Path("/home/nathan/Expedition Competition")
CW = ROOT / "Codex Work"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(CW)]

from ae_sheet_selfadapt import attach
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc


def crop_image(row, size=224):
    im = cv2.imread(str(ROOT / "data/ae_prep/chips" / row.file), cv2.IMREAD_GRAYSCALE)
    h = size // 2; x, y = int(round(row.px)), int(round(row.py))
    out = np.zeros((size, size), np.uint8)
    x0, x1, y0, y1 = max(0, x-h), min(im.shape[1], x+h), max(0, y-h), min(im.shape[0], y+h)
    ox, oy = x0-(x-h), y0-(y-h)
    out[oy:oy+y1-y0, ox:ox+x1-x0] = im[y0:y1, x0:x1]
    return out


def prepare():
    d = pd.read_parquet(ROOT / "work/ae/oof_ae_fcos_fast_layers_boxmeta.parquet")
    d = d[(d.cls == "fortifikatsii") & (d.score >= .30)].reset_index(drop=True)
    ann = json.loads((ROOT / "data/ae_prep/ann.json").read_text())
    d = attach(d, ann, half=112).reset_index(drop=True)
    lab = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    lab = lab[(lab["mod"] == "Ae") & (lab.class_name == "fortifikatsii")]
    trees = {}
    for region, g in lab.groupby("region"):
        ge = list(g.geometry); trees[region] = (STRtree(ge), ge)
    y = []
    for r in d.itertuples():
        pt = Point(r.x, r.y); hit = False
        if r.region in trees:
            tree, ge = trees[r.region]
            hit = any(ge[int(j)].covers(pt) for j in tree.query(pt))
        y.append(int(hit))
    d["target"] = y
    d.to_parquet(CW / "ae_fort_crop_records.parquet")
    print("records", len(d), "positive", int(d.target.sum()),
          "folds", d.groupby("fold").target.agg(["count", "sum"]).to_dict("index"), flush=True)


def extract():
    import timm
    import torch
    d = pd.read_parquet(CW / "ae_fort_crop_records.parquet")
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                          dynamic_img_size=True).cuda().eval()
    mean = torch.tensor([.485, .456, .406], device="cuda")[:, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[:, None, None]
    z, hand = [], []
    with torch.inference_mode():
        for i in range(0, len(d), 64):
            ims = np.stack([crop_image(r) for r in d.iloc[i:i+64].itertuples()])
            x = torch.from_numpy(ims).cuda().float()[:, None].repeat(1, 3, 1, 1) / 255
            with torch.autocast("cuda", dtype=torch.float16):
                z.append(m(((x - mean) / std).float()).float().cpu().numpy())
            print("extract", min(i + 64, len(d)), "/", len(d), flush=True)
    for r in d.itertuples():
        im = crop_image(r).astype(np.float32) / 255
        q = [im.mean(), im.std(), *np.quantile(im, [.05, .25, .5, .75, .95])]
        for s in (16, 32, 64, 96):
            c = im[112-s:112+s, 112-s:112+s]
            q += [c.mean(), c.std(), float(cv2.Laplacian(c, cv2.CV_32F).std())]
        hand.append(q)
    np.savez_compressed(CW / "ae_fort_crop_features.npz",
                        dino=np.concatenate(z).astype(np.float16), hand=np.asarray(hand, np.float32))


def f1(v):
    return 2 * v[0] / max(2 * v[0] + v[2] + .5 * v[1], 1e-12)


def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score

    d = pd.read_parquet(CW / "ae_fort_crop_records.parquet")
    z = np.load(CW / "ae_fort_crop_features.npz")
    meta = np.c_[d[["score", "bw_m", "bh_m", "aspect", "box_area_m2", "edge_m"]].to_numpy(np.float32), z["hand"]]
    folds = sorted(d.fold.unique()); pred = {}
    for name, X in (("dino", z["dino"].astype(np.float32)), ("meta", meta),
                    ("both", np.c_[z["dino"].astype(np.float32), meta])):
        for C in (.003, .01, .03, .1):
            p = np.zeros(len(d), np.float32)
            for fold in folds:
                tr = d.fold != fold; te = ~tr
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
                m.fit(X[tr], d.target[tr]); p[te] = m.predict_proba(X[te])[:, 1]
            pred[(name, C)] = p
            print(name, C, "AUC", roc_auc_score(d.target, p), "AP", average_precision_score(d.target, p), flush=True)

    lab = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    lab = lab[(lab["mod"] == "Ae") & (lab.class_name == "fortifikatsii") & lab.region.isin(d.region.unique())]
    G = layer_gt(lab); rf = d.groupby("region").fold.first().astype(int).to_dict()
    GT = {f: to_gt_geojson(G[G.region.map(rf) == f]) for f in folds}
    settings = [("base", .4, 0.)]
    for key in pred:
        for st in (.30, .35, .40, .45, .50):
            for vt in (.3, .4, .5, .6, .7, .8):
                settings.append((key, st, vt))
    counts = {}
    for si, setting in enumerate(settings):
        if setting[0] == "base": q = d[d.score >= setting[1]]
        else:
            key, st, vt = setting; q = d[(d.score >= st) & (pred[key] >= vt)]
        for fold in folds:
            _, r = score(fc(q[q.fold == fold]), GT[fold], classes=["fortifikatsii"])
            v = r["fortifikatsii"]; counts[(si, fold)] = np.array((v["tp"], v["fp"], v["fn"]), int)
    base = {f: counts[(0, f)] for f in folds}; total = np.zeros(3, int); chosen = []
    for fold in folds:
        si = max(range(len(settings)), key=lambda j: f1(sum((counts[(j, g)] for g in folds if g != fold), np.zeros(3, int))))
        chosen.append(settings[si]); total += counts[(si, fold)]
        print("fold", fold, "choice", settings[si], "base", tuple(base[fold]), f1(base[fold]),
              "test", tuple(counts[(si, fold)]), f1(counts[(si, fold)]))
    bt = sum(base.values(), np.zeros(3, int))
    print("base pooled", tuple(bt), f1(bt)); print("nested pooled", tuple(total), f1(total), chosen)
    pooled = {i: sum((counts[(i, f)] for f in folds), np.zeros(3, int)) for i in range(len(settings))}
    for i in sorted(pooled, key=lambda j: f1(pooled[j]), reverse=True)[:10]:
        print("oracle", settings[i], tuple(pooled[i]), f1(pooled[i]))


if __name__ == "__main__":
    mode = sys.argv[1]
    globals()[mode]()
