"""Morphometric damaged / intact classifier for LiDAR kurgan detections (dual-task: generic kurgan peak + preservation).

For each OOF kurgan peak (decode_site_v2 settings) that matches a kurgan label, features from the relief channels around
the peak: local relief at 2/5/10/25 m (centre, radial means at 1–12 m, max), slope (mean / max / ring asymmetry), curvature,
centre dent (looting pit: centre vs 2 m ring), ring ditch signal, radial asymmetry (ploughing cuts), the model's own mask
probabilities. LightGBM, out-of-fold by the LiDAR folds; then the nested metric with the classifier's class probability
(dual emission when uncertain) vs the current mask-ratio rule.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
RADII = (1, 2, 3, 5, 8, 12)        # metres
ANG = np.linspace(0, 2 * np.pi, 16, endpoint=False)


def feats_at(x, prob, r0, c0, res=0.5):
    H, W = x.shape[1:]
    f = {}
    for ci, name in enumerate(("lrm2", "lrm5", "lrm10", "lrm25", "slope", "hs315", "hs45", "lap")):
        ch = x[ci]
        f[f"{name}_c"] = float(ch[r0, c0])
        prof = []
        for rad in RADII:
            rr = np.clip(np.round(r0 + rad / res * np.sin(ANG)).astype(int), 0, H - 1)
            cc = np.clip(np.round(c0 + rad / res * np.cos(ANG)).astype(int), 0, W - 1)
            ring = ch[rr, cc]
            prof.append(ring.mean())
            if name in ("lrm5", "lrm10", "slope"):
                f[f"{name}_r{rad}_std"] = float(ring.std())          # asymmetry (ploughing, cuts)
                f[f"{name}_r{rad}_min"] = float(ring.min())
        for rad, v in zip(RADII, prof):
            f[f"{name}_r{rad}"] = float(v)
    f["dent"] = f["lrm2_c"] - f["lrm2_r2"]                   # summit pit (looting) → negative
    f["ditch"] = f["lrm5_r8"] - f["lrm5_r12"]                 # ring ditch
    f["height5"] = f["lrm5_c"] - f["lrm5_r12"]
    f["height10"] = f["lrm10_c"] - f["lrm10_r12"]
    y0, y1, x0, x1 = max(0, r0 - 2), r0 + 3, max(0, c0 - 2), c0 + 3
    f["p_dam"] = float(prob[3][y0:y1, x0:x1].mean()); f["p_int"] = float(prob[4][y0:y1, x0:x1].mean())
    f["p_ratio"] = f["p_dam"] / (f["p_dam"] + f["p_int"] + 1e-6)
    f["peak"] = float(prob[0][r0, c0])
    return f


def main():
    import lightgbm as lgb
    from pyproj import Transformer
    from shapely.geometry import Point
    from koz3.li_train import labels, site_items, decode_site_v2, CLS, N_HM
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    from koz3.score import score
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = pickle.load(open(os.path.join(ROOT, "work/li/oof.pkl"), "rb"))
    rows = []
    for r, p in P.items():
        it = items[r]; meta = dict(it["meta"], region=r); p = p.astype(np.float32)
        x = np.asarray(it["x"], np.float32)
        feats = decode_site_v2(p, it["valid"], meta, thr=0.1, dual=1.1)       # one emission per peak
        tr = Transformer.from_crs(3857, meta["crs"], always_xy=True)
        lab = L[(L.region == r) & L.class_name.str.startswith("kurgany")].to_crs(meta["crs"])
        geoms, cls = list(lab.geometry), list(lab.class_name)
        for f in feats:
            if not f["properties"]["class_name"].startswith("kurgany"):
                continue
            from shapely.geometry import shape
            c = shape(f["geometry"]).centroid; X, Y = tr.transform(c.x, c.y)
            col, row = int((X - meta["x0"]) / meta["res"]), int((meta["ytop"] - Y) / meta["res"])
            if not (0 <= row < x.shape[1] and 0 <= col < x.shape[2]):
                continue
            hit = [k for k, g in zip(cls, geoms) if g.contains(Point(X, Y))]
            d = feats_at(x, p, row, col)
            d.update(region=r, fold=it["fold"], x=c.x, y=c.y, conf=f["properties"]["confidence"],
                     y_dam=(1 if hit and hit[0] == "kurgany_povrezhdennye" else 0) if hit else -1)
            rows.append(d)
    D = pd.DataFrame(rows)
    print(len(D), "kurgan peaks;", int((D.y_dam >= 0).sum()), "on a kurgan label")
    feat_cols = [c for c in D.columns if c not in ("region", "fold", "x", "y", "conf", "y_dam")]
    D["p_cls"] = np.nan
    for f in sorted(D.fold.unique()):
        tr_ = D[(D.fold != f) & (D.y_dam >= 0)]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=0, verbose=-1)
        m.fit(tr_[feat_cols], tr_.y_dam)
        D.loc[D.fold == f, "p_cls"] = m.predict_proba(D.loc[D.fold == f, feat_cols])[:, 1]
    lab_ = D[D.y_dam >= 0]
    acc_rule = ((lab_.p_ratio >= 0.5).astype(int) == lab_.y_dam).mean(); acc_cls = ((lab_.p_cls >= 0.5).astype(int) == lab_.y_dam).mean()
    from sklearn.metrics import roc_auc_score
    print(f"damaged-vs-intact on matched peaks: mask-ratio rule acc {acc_rule:.3f} AUC {roc_auc_score(lab_.y_dam, lab_.p_ratio):.3f} | "
          f"LightGBM OOF acc {acc_cls:.3f} AUC {roc_auc_score(lab_.y_dam, lab_.p_cls):.3f}")
    D.to_parquet(os.path.join(ROOT, "work/li/kurgan_cls_oof.parquet"))
    # nested metric: replace the kurgan peaks' classes; other LiDAR classes unchanged
    folds = sorted(D.fold.unique())
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if items[r]["fold"] == f])]) for f in folds}
    others = {}
    for r, p in P.items():
        it = items[r]
        others[r] = [f for f in decode_site_v2(p.astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=0.1, dual=0.3)
                     if not f["properties"]["class_name"].startswith("kurgany")]
    kk = lambda yy: 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(yy / 6378137.0)) - np.pi / 2)))

    def make(s, f):
        src, dual = s
        out = [g for r in P if items[r]["fold"] == f for g in others[r]]
        for row in D[D.fold == f].itertuples():
            pd_ = row.p_cls if src == "cls" else row.p_ratio
            emit = [("kurgany_povrezhdennye", pd_), ("kurgany_tselye", 1 - pd_)]
            emit.sort(key=lambda e: -e[1])
            emit = [emit[0]] + [e for e in emit[1:] if e[1] >= dual]
            for c, pc in emit:
                out.append({"type": "Feature", "geometry": Point(row.x, row.y).buffer(kk(row.y), 12).__geo_interface__,
                            "properties": {"class_name": c, "confidence": float(row.conf * pc)}})
        return {"type": "FeatureCollection", "features": out}
    S = [(src, d) for src in ("rule", "cls") for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1.1)]
    C = fold_counts(make, S, folds, gt, CLS)
    for src in ("rule", "cls"):
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[0] == src})
        print(f"{src:4s} honest {h:.4f} (oracle {o:.4f} @ {os_}) " + " ".join(f"{c[:7]} {2*v[0]/max(2*v[0]+v[2]+0.5*v[1],1):.3f}" for c, v in tot.items() if v[0] + v[2]))


if __name__ == "__main__":
    main()
