#!/usr/bin/env python3
"""Fold-honest aerial scene gate using only observable image metadata.

Extends the failed proposal-statistics gate with source raster geometry and
photometric quality measures.  The gate and per-scene cap are selected inside
each geographic outer fold and evaluated with source-image-aware references.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path("/home/nathan/Expedition Competition")
CW = ROOT / "Codex Work"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(CW)]
from ae_layer_oof import layer_gt
from ae_scene_gate import scene_features
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

KP = "kurgany_povrezhdennye"
CAPS = (0, 5, 10, 20, 50, 100, 200)
GATE_Q = (0.0, .15, .30, .45, .60, .75, .90)


def f1(c):
    tp, fp, fn = c
    return 2*tp / max(2*tp + fn + .5*fp, 1e-12)


def counts(pred, gt):
    _, r = score(fc(pred), gt, classes=[KP]); v = r[KP]
    return np.array((v["tp"], v["fp"], v["fn"]), int)


def raster_features(region, scene):
    paths = glob.glob(str(ROOT / "data/train-expds-3" / region / "*_Ae_*" / f"{scene}.tif"))
    if not paths:
        paths = glob.glob(str(ROOT / "data/train-expds-3" / region / "*_Ae_*" / "*.tif"))
        paths = [p for p in paths if Path(p).stem == scene]
    if len(paths) != 1:
        return {}
    p = paths[0]
    with rasterio.open(p) as src:
        h, w = src.height, src.width
        sy = min(512, h); sx = min(512, w)
        im = src.read(1, out_shape=(sy, sx), masked=True).filled(0).astype(np.float32)
        gsdx, gsdy = abs(src.transform.a), abs(src.transform.e)
    x = im[im != 0]
    if len(x) < 100:
        return {}
    p2, p10, p50, p90, p98 = np.percentile(x, (2,10,50,90,98))
    z = np.clip((im-p2) * 255/max(p98-p2, 1e-6), 0, 255).astype(np.uint8)
    valid = im != 0
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)[valid]
    lap = cv2.Laplacian(z, cv2.CV_32F)[valid]
    hist = np.histogram(z[valid], bins=16, range=(0,256))[0].astype(float); hist /= hist.sum()
    entropy = -np.sum(hist[hist>0] * np.log2(hist[hist>0]))
    return {
        "r_width": w, "r_height": h, "r_log_pixels": np.log1p(w*h),
        "r_aspect": max(w,h)/max(min(w,h),1), "r_gsdx": gsdx, "r_gsdy": gsdy,
        "r_area": w*h*gsdx*gsdy, "r_valid": valid.mean(),
        "r_p50": p50, "r_spread": p98-p2, "r_iqr": p90-p10,
        "r_entropy": entropy, "r_grad_mean": mag.mean(), "r_grad90": np.percentile(mag,90),
        "r_lap_std": lap.std(), "r_aniso": abs(np.mean(abs(gx[valid]))-np.mean(abs(gy[valid]))),
    }


def model(kind, seed):
    if kind == "logit":
        return make_pipeline(SimpleImputer(), RobustScaler(), LogisticRegression(
            C=.15, class_weight="balanced", max_iter=2000, random_state=seed))
    return make_pipeline(SimpleImputer(), ExtraTreesClassifier(
        n_estimators=500, max_depth=2, min_samples_leaf=3, class_weight="balanced",
        max_features=.65, random_state=seed, n_jobs=-1))


def pred(kind, Xtr, ytr, Xte, seed):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr[0]) if len(ytr) else 0.)
    m = model(kind, seed); m.fit(Xtr, ytr); return m.predict_proba(Xte)[:,1]


def main():
    d = pd.read_parquet(CW / "ae_sheet_selfadapt_features.parquet")
    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    labels = labels[(labels["mod"] == "Ae") & labels.region.isin(d.region.unique())]
    layered = layer_gt(labels)
    fold_of = d.groupby("region").fold.first().astype(int).to_dict(); folds = sorted(set(fold_of.values()))
    byscene, scene_gt, rows = {}, {}, []
    for (region, scene), g in d.groupby(["region","scene"]):
        g = g.sort_values("score", ascending=False).copy(); byscene[(region,scene)] = g
        gg = layered[(layered.region==region)&(layered.scene==scene)]
        scene_gt[(region,scene)] = to_gt_geojson(gg)
        q = g.head(100).copy(); q["cls"] = KP; c = counts(q, scene_gt[(region,scene)])
        rows.append({"region":region,"scene":scene,"fold":fold_of[region],
                     **scene_features(g), **raster_features(region,scene),
                     "tp100":c[0],"fp100":c[1],"fn100":c[2],"useful":int(c[0]>0)})
    s = pd.DataFrame(rows).sort_values(["fold","region","scene"]).reset_index(drop=True)
    detector = [c for c in s if c not in {"region","scene","fold","tp100","fp100","fn100","useful"} and not c.startswith("r_")]
    image = [c for c in s if c.startswith("r_")]
    feature_sets = {"det":detector, "image":image, "both":detector+image}
    gt_fold = {f:to_gt_geojson(layered[layered.region.map(fold_of)==f]) for f in folds}
    total=np.zeros(3,int); choices=[]; selected=[]
    for outer in folds:
        train_folds=[f for f in folds if f!=outer]
        inner={}
        for fs,cols in feature_sets.items():
            for kind in ("logit","trees"):
                z=np.full(len(s),np.nan)
                for inner_fold in train_folds:
                    tr=s.fold.isin([f for f in train_folds if f!=inner_fold]); va=s.fold==inner_fold
                    z[va]=pred(kind,s.loc[tr,cols],s.loc[tr,"useful"],s.loc[va,cols],1000+10*outer+inner_fold)
                inner[(fs,kind)]=z
        candidates=[]
        for (fs,kind),z in inner.items():
            vals=z[s.fold.isin(train_folds)]
            thrs=sorted(set(np.quantile(vals,GATE_Q).tolist()+[-1.,1.1]))
            for cap in CAPS:
                for th in thrs:
                    c=np.zeros(3,int)
                    for f in train_folds:
                        pred_parts=[]
                        for row in s[(s.fold==f)&(z>=th)].itertuples():
                            q=byscene[(row.region,row.scene)].head(cap).copy();q["cls"]=KP;pred_parts.append(q)
                        pp=pd.concat(pred_parts,ignore_index=True) if pred_parts else d.iloc[:0].copy()
                        c+=counts(pp,gt_fold[f])
                    candidates.append((f1(c),fs,kind,cap,th,c))
        _,fs,kind,cap,th,ic=max(candidates,key=lambda z:(z[0],-z[3]))
        cols=feature_sets[fs];tr=s.fold.isin(train_folds);te=s.fold==outer
        z=pred(kind,s.loc[tr,cols],s.loc[tr,"useful"],s.loc[te,cols],2000+outer)
        parts=[]; kept=[]
        for row,p in zip(s[te].itertuples(),z):
            if p>=th and cap:
                q=byscene[(row.region,row.scene)].head(cap).copy();q["cls"]=KP;parts.append(q);kept.append((row.region,row.scene,float(p)))
        pp=pd.concat(parts,ignore_index=True) if parts else d.iloc[:0].copy(); c=counts(pp,gt_fold[outer]);total+=c
        choices.append((outer,fs,kind,cap,round(float(th),4),tuple(ic),tuple(c),len(kept)));selected.extend((outer,*x) for x in kept)
    print("scenes",len(s),"useful",int(s.useful.sum()),"features",{k:len(v) for k,v in feature_sets.items()})
    print("nested",f1(total),tuple(total))
    for x in choices:print("choice",x)
    print("selected");[print(x) for x in selected]
    s.to_csv(CW / "ae_scene_metadata_table.csv",index=False)


if __name__ == "__main__":
    main()
