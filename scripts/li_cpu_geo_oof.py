"""CPU geomorphometric kurgan pipeline (parallel over sites), nested over the LiDAR site folds:
  candidates  multi-scale Laplacian-of-Gaussian peaks on the positive 10 m local-relief model (σ 2–12 px = 1–6 m)
  features    relief rings / slope / curvature (li_kurgan_cls.feats_at), LoG scale + response, rotation-invariant
              angular-harmonic magnitudes (k = 1–4) at 3 radii (≈ Zernike-style shape descriptors), Gaussian profile fit
              R² (N–S and E–W), radial gradient symmetry, half-height region compactness / eccentricity / solidity,
              neighbour counts; optional CNN heat / mask features
  tier 1      LightGBM "best candidate of a kurgan label" (other folds only)
  tier 2      damaged / intact from the CNN mask ratio; hedge when |p_dam − p_int| < 0.15·(p_dam + p_int)
Modes: ADD (candidates > 4 m from every CNN kurgan peak, added to the deployed CNN decoding) and REPLACE.
  python scripts/li_cpu_geo_oof.py [work/li/oof_augx.pkl]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")
ANG = np.linspace(0, 2 * np.pi, 32, endpoint=False)


def site_cands(args):
    """One site: LoG candidates + features + labels (runs in a worker)."""
    r, oof_path = args
    from scipy.ndimage import gaussian_filter, sobel
    from scipy.spatial import cKDTree
    from skimage.feature import blob_log
    from skimage.measure import label as sklabel, regionprops
    from shapely import STRtree
    from shapely.geometry import Point, shape
    from pyproj import Transformer
    from koz3.li_train import labels, site_items, decode_site_v2
    from li_kurgan_cls import feats_at
    L = labels(); it = {i["region"]: i for i in site_items(L)}[r]
    P = SITE_P[r].astype(np.float32); x = np.asarray(it["x"], np.float32); meta = it["meta"]; valid = it["valid"]
    lrm = np.clip(gaussian_filter(x[2], 1.0), 0, None)
    blobs = blob_log(lrm, min_sigma=2, max_sigma=12, num_sigma=6, threshold=0.05, overlap=0.5)
    blobs = blobs[[valid[int(b[0]), int(b[1])] for b in blobs]] if len(blobs) else blobs
    if len(blobs) > 6000:
        o = np.argsort(-lrm[blobs[:, 0].astype(int), blobs[:, 1].astype(int)])[:6000]; blobs = blobs[o]
    H, W = lrm.shape; gx, gy = sobel(x[2], 1), sobel(x[2], 0)
    tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True)
    dep = decode_site_v2(P, valid, dict(meta, region=r), thr=0.2, dual=0.05)
    cnn = np.array([shape(f["geometry"]).centroid.coords[0] for f in dep if f["properties"]["class_name"] in K]).reshape(-1, 2)
    lab = L[(L.region == r) & L.class_name.isin(K)].to_crs(meta["crs"]); geoms = list(lab.geometry); tree = STRtree(geoms) if geoms else None
    rows, best = [], {}
    for yy, xx, sg in blobs:
        yi, xi = int(yy), int(xx)
        if not (14 <= yi < H - 14 and 14 <= xi < W - 14):
            continue
        d = feats_at(x, P, yi, xi)
        d["log_sigma"] = float(sg); d["log_resp"] = float(lrm[yi, xi])
        for rad in (3, 6, 10):                                     # angular harmonics (rotation-invariant magnitudes)
            ring = x[2][np.clip(np.round(yi + rad * np.sin(ANG)).astype(int), 0, H - 1), np.clip(np.round(xi + rad * np.cos(ANG)).astype(int), 0, W - 1)]
            spec = np.abs(np.fft.rfft(ring - ring.mean())) / (np.abs(ring).mean() + 1e-3)
            for k in range(1, 5):
                d[f"harm_r{rad}_k{k}"] = float(spec[k])
        R = int(max(6, min(14, 3 * sg)))
        tmpl = np.exp(-(np.arange(-R, R + 1) ** 2) / (2 * max(sg, 1.5) ** 2))
        for nm, prof in (("ns", x[2][yi - R:yi + R + 1, xi]), ("ew", x[2][yi, xi - R:xi + R + 1])):
            A = np.c_[tmpl, np.ones_like(tmpl)]; coef, res, *_ = np.linalg.lstsq(A, prof, rcond=None)
            ss = ((prof - prof.mean()) ** 2).sum() + 1e-9
            d[f"gfit_r2_{nm}"] = float(1 - ((prof - A @ coef) ** 2).sum() / ss); d[f"gfit_amp_{nm}"] = float(coef[0])
        py, px = np.mgrid[-R:R + 1, -R:R + 1]; dist = np.hypot(py, px) + 1e-6
        gpy, gpx = gy[yi - R:yi + R + 1, xi - R:xi + R + 1], gx[yi - R:yi + R + 1, xi - R:xi + R + 1]
        mag = np.hypot(gpy, gpx) + 1e-6
        d["radial_sym"] = float(((-(gpy * py + gpx * px) / dist) / mag * mag).sum() / mag.sum())   # gradients pointing inward
        patch = x[2][yi - R:yi + R + 1, xi - R:xi + R + 1]; half = patch > (patch[R, R] + patch.min()) / 2
        lb = sklabel(half); k0 = lb[R, R]
        if k0:
            rp = regionprops((lb == k0).astype(int))[0]
            d["compact"] = float(4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2); d["ecc"] = float(rp.eccentricity); d["solidity"] = float(rp.solidity)
        else:
            d["compact"] = d["ecc"] = d["solidity"] = np.nan
        X_, Y_ = meta["x0"] + (xi + 0.5) * meta["res"], meta["ytop"] - (yi + 0.5) * meta["res"]
        mx, my = tr.transform(X_, Y_)
        d.update(region=r, fold=it["fold"], x=mx, y=my, lab=0,
                 d_cnn=float(cKDTree(cnn).query((mx, my))[0]) if len(cnn) else 1e9)
        if tree is not None:
            pt = Point(X_, Y_)
            for j in tree.query(pt):
                if geoms[j].contains(pt) and d["log_resp"] > best.get(j, (-1, -1))[0]:
                    best[j] = (d["log_resp"], len(rows))
        rows.append(d)
    for _, i in best.values():
        rows[i]["lab"] = 1
    D = pd.DataFrame(rows)
    if len(D):
        from scipy.spatial import cKDTree as KD
        t = KD(D[["x", "y"]].values)
        for Rm in (10, 25, 60):
            D[f"nb{Rm}"] = [len(n) - 1 for n in t.query_ball_point(D[["x", "y"]].values, Rm)]
    return D


SITE_P = {}


def main(oof="work/li/oof_augx.pkl"):
    global SITE_P
    import lightgbm as lgb
    from joblib import Parallel, delayed
    from shapely.geometry import Point
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    SITE_P = pickle.load(open(os.path.join(ROOT, oof), "rb"))
    cache = os.path.join(ROOT, "work/li/cpu_geo_cands.parquet")
    if os.path.exists(cache):
        D = pd.read_parquet(cache)
    else:
        parts = Parallel(n_jobs=12, backend="loky")(delayed(site_cands)((r, oof)) for r in sorted(SITE_P))
        D = pd.concat([p for p in parts if len(p)], ignore_index=True); D.to_parquet(cache)
    print(len(D), "LoG candidates;", int(D.lab.sum()), "best matches of kurgan labels")
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    meta_cols = ("region", "fold", "x", "y", "lab", "d_cnn")
    phys = [c for c in D.columns if c not in meta_cols and not c.startswith("p_") and c != "peak"]
    withcnn = phys + ["p_dam", "p_int", "p_ratio", "peak"]
    from sklearn.metrics import roc_auc_score
    for nm, cols in (("geo", phys), ("geocnn", withcnn)):
        D[f"q_{nm}"] = np.nan
        for f in sorted(D.fold.unique()):
            tr_ = D[D.fold != f]
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=12)
            m.fit(tr_[cols], tr_.lab)
            D.loc[D.fold == f, f"q_{nm}"] = m.predict_proba(D.loc[D.fold == f, cols])[:, 1]
        print(f"tier-1 {nm:7s} AUC {roc_auc_score(D.lab, D[f'q_{nm}']):.3f}  (CNN heat on the same candidates {roc_auc_score(D.lab, D.peak):.3f})")
    folds = sorted(D.fold.unique())
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in SITE_P if items[r]["fold"] == f])]) for f in folds}
    dep = {r: decode_site_v2(SITE_P[r].astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=0.2, dual=0.05) for r in SITE_P}
    kk = lambda yy: 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(yy / 6378137.0)) - np.pi / 2)))

    def emit(G, src):
        out = []
        for row in G.itertuples():
            pd_, pi_ = row.p_dam, row.p_int; tot = pd_ + pi_ + 1e-6
            e = [("kurgany_povrezhdennye", pd_ / tot), ("kurgany_tselye", pi_ / tot)]; e.sort(key=lambda z: -z[1])
            if abs(pd_ - pi_) < 0.15 * tot:                          # tier-2 hedge
                pass
            else:
                e = e[:1]
            for c, pc in e:
                out.append({"type": "Feature", "geometry": Point(row.x, row.y).buffer(kk(row.y), 12).__geo_interface__,
                            "properties": {"class_name": c, "confidence": float(getattr(row, src) * pc)}})
        return out

    def make(s, f):
        mode, src, t = s
        base = [g for r in SITE_P if items[r]["fold"] == f for g in dep[r]]
        if mode == "dep":
            return {"type": "FeatureCollection", "features": base}
        if mode == "add":
            return {"type": "FeatureCollection", "features": base + emit(D[(D.fold == f) & (D[src] >= t) & (D.d_cnn > 4.0)], src)}
        keep = [g for g in base if g["properties"]["class_name"] not in K]
        return {"type": "FeatureCollection", "features": keep + emit(D[(D.fold == f) & (D[src] >= t)], src)}
    S = [("dep", "", 0)] + [(m, s, t) for m in ("add",) for s in ("q_geo", "q_geocnn") for t in (0.5, 0.6, 0.7, 0.8, 0.9)] \
        + [("rep", s, t) for s in ("q_geo", "q_geocnn") for t in (0.1, 0.15, 0.2, 0.3, 0.4)]
    C = fold_counts(make, S, folds, gt, CLS)
    for key in (("dep", ""), ("add", "q_geo"), ("add", "q_geocnn"), ("rep", "q_geo"), ("rep", "q_geocnn")):
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[:2] == key})
        print(f"{key[0]:3s} {key[1]:9s} honest {h:.4f} (oracle {o:.4f} @ {os_[2]}) " + " ".join(f"{c[:8]} {2*v[0]/max(2*v[0]+v[2]+0.5*v[1],1):.3f}" for c, v in tot.items() if v[0] + v[2]))


if __name__ == "__main__":
    main(*sys.argv[1:])
