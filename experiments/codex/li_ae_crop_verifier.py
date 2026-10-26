#!/usr/bin/env python3
"""Image verifier at LiDAR mound coordinates for repeated aerial observations.

The detector never scans arbitrary film background.  It asks a narrower,
deployment-aligned question: does the aerial crop at an independently detected
LiDAR mound contain an aerial kurgan observation?  Feature extraction is frozen
DINOv2; the verifier is cross-fitted by geographic fold.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path("/home/nathan/Expedition Competition")
CW = ROOT / "Codex Work"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def prepare():
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree
    from koz3.li_train import labels, site_items, decode_site_v2
    import li_compare_oof as lc

    ann = json.load(open(ROOT / "data/ae_prep/ann.json"))
    by_region = {}
    for c in ann:
        c = dict(c)
        c["inv"] = ~Affine(*c["transform"])
        c["tr"] = Transformer.from_crs(3857, c["crs"], always_xy=True)
        by_region.setdefault(c["region"], []).append(c)

    lab = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    lab = lab[(lab["mod"] == "Ae") & lab.class_name.isin(K) & lab.geometry.notna()].copy()
    gt = {}
    for r, g in lab.groupby("region"):
        ge = list(g.geometry); gt[r] = (STRtree(ge), ge, list(g.class_name))

    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(str(ROOT / f"work/li/oof_{nm}.pkl.zst")) for nm in ("ens", "aug", "augx")}
    prior = {"R": 200.0, "K": 3, "snb": 0.2}; rows = []
    for r in sorted(set(P["ens"]) & set(by_region)):
        it = items[r]
        p = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32)
        feats = decode_site_v2(p, it["valid"], dict(it["meta"], region=r), thr=.15, dual=.05, comp_thr=.5, prior=prior)
        for f in feats:
            pr = f["properties"]
            if pr["class_name"] not in K:
                continue
            q = shape(f["geometry"]).centroid
            choices = []
            for c in by_region[r]:
                X, Y = c["tr"].transform(q.x, q.y); x, y = c["inv"] * (X, Y)
                if 0 <= x < c["w"] and 0 <= y < c["h"]:
                    margin = min(x, y, c["w"] - x, c["h"] - y)
                    choices.append((margin, c, x, y))
            if not choices:
                continue
            _, c, x, y = max(choices, key=lambda z: z[0])
            hit_any = hit_same = 0
            if r in gt:
                tree, ge, classes = gt[r]
                for j in tree.query(Point(q.x, q.y)):
                    if ge[int(j)].covers(Point(q.x, q.y)):
                        hit_any = 1
                        hit_same |= int(classes[int(j)] == pr["class_name"])
            rows.append(dict(region=r, fold=it["fold"], cls=pr["class_name"], li_score=pr["confidence"],
                             x3857=q.x, y3857=q.y, file=c["file"], px=x, py=y,
                             target_any=hit_any, target_same=hit_same))
    d = pd.DataFrame(rows)
    d.to_parquet(CW / "li_ae_crop_records.parquet")
    print("records", len(d), "sites", d.region.nunique(), "any positive", int(d.target_any.sum()),
          "same positive", int(d.target_same.sum()), "by fold", d.groupby("fold").target_any.agg(["count", "sum"]).to_dict("index"))


def crop_image(row, size=224):
    p = ROOT / "data/ae_prep/chips" / row.file
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    h = size // 2; x, y = int(round(row.px)), int(round(row.py))
    out = np.zeros((size, size), np.uint8)
    x0, x1, y0, y1 = max(0, x-h), min(im.shape[1], x+h), max(0, y-h), min(im.shape[0], y+h)
    ox, oy = x0-(x-h), y0-(y-h)
    out[oy:oy+y1-y0, ox:ox+x1-x0] = im[y0:y1, x0:x1]
    return out


def extract():
    import torch
    import timm
    d = pd.read_parquet(CW / "li_ae_crop_records.parquet")
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                          dynamic_img_size=True).cuda().eval()
    mean = torch.tensor([.485, .456, .406], device="cuda")[:, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[:, None, None]
    feats = []
    with torch.no_grad():
        for i in range(0, len(d), 64):
            ims = np.stack([crop_image(r) for r in d.iloc[i:i+64].itertuples()])
            x = torch.from_numpy(ims).cuda().float()[:, None].repeat(1, 3, 1, 1) / 255
            x = (x - mean) / std
            with torch.autocast("cuda", dtype=torch.float16): z = m(x)
            feats.append(z.float().cpu().numpy())
            print("extract", min(i+64, len(d)), len(d), flush=True)
    z = np.concatenate(feats)
    # Compact photometric/context descriptors remain available to the linear head.
    hand = []
    for r in d.itertuples():
        im = crop_image(r).astype(np.float32) / 255
        q = [im.mean(), im.std(), *np.quantile(im, [.05, .25, .5, .75, .95])]
        for s in (16, 32, 64):
            c = im[112-s:112+s, 112-s:112+s]
            q += [c.mean(), c.std(), float(cv2.Laplacian(c, cv2.CV_32F).std())]
        hand.append(q)
    np.savez_compressed(CW / "li_ae_crop_features.npz", dino=z.astype(np.float16), hand=np.asarray(hand, np.float32))
    print("saved", z.shape)


def evaluate():
    from shapely.geometry import shape
    from sklearn.ensemble import RandomForestClassifier
    from scipy.spatial import cKDTree
    from koz3.li_train import labels, site_items, decode_site_v2
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    import li_compare_oof as lc

    d = pd.read_parquet(CW / "li_ae_crop_records.parquet")
    z = np.load(CW / "li_ae_crop_features.npz")
    ext = pd.read_parquet(ROOT / "work/ae/sheet_extent.parquet")
    d = d.join(ext[["cx", "cy", "half"]], on="region")
    d["drel"] = np.hypot(d.x3857-d.cx, d.y3857-d.cy)/d["half"]
    d["region_n"] = d.groupby("region").region.transform("size")
    for rad in (25, 50, 100, 200, 500):
        d[f"n{rad}"] = 0
        for r, g in d.groupby("region"):
            tree = cKDTree(g[["x3857", "y3857"]])
            scale = 1/np.cos(np.radians(np.degrees(2*np.arctan(np.exp(g.y3857.mean()/6378137))-np.pi/2)))
            d.loc[g.index, f"n{rad}"] = [len(tree.query_ball_point(p,rad*scale)) for p in g[["x3857","y3857"]].values]
    meta = d[["li_score","drel","region_n","n25","n50","n100","n200","n500"]].to_numpy(np.float32)
    X = np.c_[z["hand"], meta, (d.cls == K[0]).to_numpy()]
    score = np.zeros(len(d), np.float32)
    folds = sorted(set(d.fold) - {1})
    for f in folds:
        tr = (d.fold != f) & (d.fold != 1); va = d.fold == f
        m = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=10,
                                   class_weight="balanced", max_features=.8, random_state=20261026, n_jobs=-1)
        m.fit(X[tr], d.target_any[tr]); score[va] = m.predict_proba(X[va])[:, 1]
        print("fold", f, "train", int(tr.sum()), "valid", int(va.sum()))
    d["verifier"] = score
    d.to_parquet(CW / "li_ae_crop_oof.parquet")

    # Reconstruct the exact Li baseline once and add a twin only at selected rows.
    Lall = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(str(ROOT / f"work/li/oof_{nm}.pkl.zst")) for nm in ("ens", "aug", "augx")}
    prior = {"R": 200., "K": 3, "snb": .2}; base = {}
    sites = sorted(set(d.region[d.fold.isin(folds)]))
    for r in sites:
        it=items[r]; p=((P["ens"][r].astype(np.float32)+P["aug"][r]+P["augx"][r])/3).astype(np.float32)
        base[r]=decode_site_v2(p,it["valid"],dict(it["meta"],region=r),thr=.15,dual=.05,comp_thr=.5,prior=prior)
    settings=[("base",)]+[("verify", t, l) for t in (.3,.4,.5,.6,.7,.8) for l in (.25,.4)]
    def make(st, f):
        regs=[r for r in sites if items[r]["fold"]==f]; fs=[x for r in regs for x in base[r]]
        if st[0]=="base": return {"type":"FeatureCollection","features":fs}
        _,thr,lthr=st
        chosen=d[(d.fold==f)&(d.verifier>=thr)&(d.li_score>=lthr)]
        by={(r.region,r.cls,round(r.x3857,2),round(r.y3857,2)) for r in chosen.itertuples()}
        add=[]
        for x in fs:
            p=x["properties"]; q=shape(x["geometry"]).centroid
            if (p["region_name"],p["class_name"],round(q.x,2),round(q.y,2)) in by:
                t=copy.deepcopy(x);t["properties"]["markup_type"]="ae";t["properties"]["confidence"]*=.1;add.append(t)
        return {"type":"FeatureCollection","features":fs+add}
    gtframe=Lall[Lall.region.isin(sites)&Lall["mod"].isin(["Li","Ae"])]
    gt={f:to_gt_geojson(gtframe[gtframe.region.isin([r for r in sites if items[r]["fold"]==f])]) for f in folds}
    C=fold_counts(make,settings,folds,gt); b=weighted(_add_all(C[("base",)].values())); h,ch,o,os_,_=nested(C)
    print("base",b,"nested",h,"oracle",o,os_,"chosen",list(ch.values()))
    for st in settings:
        v=weighted(_add_all(C[st].values()));print(st,v,v-b)


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["prepare","extract","evaluate"]);a=ap.parse_args()
    globals()[a.mode]()
