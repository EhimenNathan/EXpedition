"""Topological + scattering features for LiDAR kurgan candidates (CPU, parallel), nested over the site folds:
  persistent homology (GUDHI cubical complex on the 64×64 local-relief patch): H0 persistence of the component born at the
  candidate (superlevel set = mound prominence), top-3 H0 persistences, H1 count / total / max persistence of the
  sublevel set (ring ditches); wavelet scattering (Kymatio, J=3, L=8) on the 32×32 patch, spatially averaged.
Same candidates (multi-scale LoG, top 3000 per site), labels, tiers and metric protocol as li_cpu_geo_oof.py.
  python scripts/li_tda_scat_oof.py work/li/oof_augx.pkl
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import li_cpu_geo_oof as G  # noqa: E402


def tda_scat(args):
    r, P = args
    import gudhi
    from kymatio.numpy import Scattering2D
    G.SITE_P = {r: P}
    D = G.site_cands((r, None))
    if not len(D):
        return D
    D = D.sort_values("log_resp", ascending=False).head(3000).reset_index(drop=True)
    from koz3.li_train import labels, site_items
    it = {i["region"]: i for i in site_items(labels())}[r]; x = np.asarray(it["x"], np.float32)[2]; meta = it["meta"]
    from pyproj import Transformer
    tr = Transformer.from_crs(3857, meta["crs"], always_xy=True)
    S = Scattering2D(J=3, shape=(32, 32), L=8)
    H, W = x.shape; feats = []
    for row in D.itertuples():
        X, Y = tr.transform(row.x, row.y)
        ci, ri = int((X - meta["x0"]) / meta["res"]), int((meta["ytop"] - Y) / meta["res"])
        f = {}
        y0, x0 = max(0, ri - 32), max(0, ci - 32); patch = x[y0:ri + 32, x0:ci + 32].astype(np.float64)
        if patch.size < 100:
            feats.append(f); continue
        cc = gudhi.CubicalComplex(top_dimensional_cells=-patch); cc.compute_persistence()
        p0 = np.array([d - b if np.isfinite(d) else patch.max() - patch.min() for b, d in cc.persistence_intervals_in_dimension(0)])
        f["h0_top1"], f["h0_top2"], f["h0_top3"] = (sorted(p0, reverse=True) + [0, 0, 0])[:3]
        try:                                   # component born at the candidate pixel
            pairs = cc.cofaces_of_persistence_pairs()[0]
            idx_c = (ri - y0) * patch.shape[1] + (ci - x0)
            loc = {int(b): (d_) for b, d_ in (pairs[0] if len(pairs) else [])}
            fin = cc.cofaces_of_persistence_pairs()[1]
            vals = -patch.ravel()
            pers = {int(b): vals[int(d_)] - vals[int(b)] for b, d_ in (pairs[0] if len(pairs) else [])}
            near = [pers[k] for k in pers if abs(k // patch.shape[1] - (ri - y0)) <= 2 and abs(k % patch.shape[1] - (ci - x0)) <= 2]
            f["h0_cand"] = float(max(near)) if near else 0.0
        except Exception:
            f["h0_cand"] = np.nan
        cs = gudhi.CubicalComplex(top_dimensional_cells=patch); cs.compute_persistence()
        p1 = np.array([d - b for b, d in cs.persistence_intervals_in_dimension(1) if np.isfinite(d)])
        f["h1_n"] = int((p1 > 0.05).sum()); f["h1_sum"] = float(p1.sum()); f["h1_max"] = float(p1.max()) if len(p1) else 0.0
        q = x[max(0, ri - 16):ri + 16, max(0, ci - 16):ci + 16]
        if q.shape == (32, 32):
            sc = S(q.astype(np.float32)).mean(axis=(-1, -2))
            for k, v in enumerate(sc):
                f[f"scat{k}"] = float(v)
        feats.append(f)
    return pd.concat([D, pd.DataFrame(feats)], axis=1)


def main(oof="work/li/oof_augx.pkl"):
    import lightgbm as lgb
    from joblib import Parallel, delayed
    from shapely.geometry import Point
    from sklearn.metrics import roc_auc_score
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    P = pickle.load(open(os.path.join(ROOT, oof), "rb"))
    cache = os.path.join(ROOT, "work/li/tda_scat_cands.parquet")
    if os.path.exists(cache):
        D = pd.read_parquet(cache)
    else:
        parts = Parallel(n_jobs=16, backend="loky")(delayed(tda_scat)((r, P[r])) for r in sorted(P))
        D = pd.concat([p for p in parts if len(p)], ignore_index=True); D.to_parquet(cache)
    print(len(D), "candidates;", int(D.lab.sum()), "best matches of kurgan labels")
    meta_cols = ("region", "fold", "x", "y", "lab", "d_cnn")
    tda = [c for c in D.columns if c.startswith("h0_") or c.startswith("h1_")]
    scat = [c for c in D.columns if c.startswith("scat")]
    geo = [c for c in D.columns if c not in meta_cols and c not in tda and c not in scat and not c.startswith("p_") and c != "peak"]
    cnn = ["p_dam", "p_int", "p_ratio", "peak"]
    sets = {"tda": tda, "scat": scat, "tda+scat": tda + scat, "geo+tda+scat": geo + tda + scat, "all+cnn": geo + tda + scat + cnn}
    for nm, cols in sets.items():
        D[f"q_{nm}"] = np.nan
        for f in sorted(D.fold.unique()):
            tr_ = D[D.fold != f]
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, random_state=0, verbose=-1, n_jobs=16).fit(tr_[cols], tr_.lab)
            D.loc[D.fold == f, f"q_{nm}"] = m.predict_proba(D.loc[D.fold == f, cols])[:, 1]
        print(f"tier-1 {nm:13s} ({len(cols):3d} feats) AUC {roc_auc_score(D.lab, D[f'q_{nm}']):.3f}")
    print(f"CNN heat on the same candidates AUC {roc_auc_score(D.lab, D.peak):.3f}")
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    folds = sorted(D.fold.unique())
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if items[r]["fold"] == f])]) for f in folds}
    dep = {r: decode_site_v2(P[r].astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=0.2, dual=0.05) for r in P}
    K = G.K
    kk = lambda yy: 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(yy / 6378137.0)) - np.pi / 2)))

    def emit(Gd, src):
        out = []
        for row in Gd.itertuples():
            tot = row.p_dam + row.p_int + 1e-6; e = [("kurgany_povrezhdennye", row.p_dam / tot), ("kurgany_tselye", row.p_int / tot)]
            e.sort(key=lambda z: -z[1]); e = e if abs(row.p_dam - row.p_int) < 0.15 * tot else e[:1]
            for c, pc in e:
                out.append({"type": "Feature", "geometry": Point(row.x, row.y).buffer(kk(row.y), 12).__geo_interface__,
                            "properties": {"class_name": c, "confidence": float(getattr(row, src) * pc)}})
        return out

    def make(s, f):
        mode, src, t = s
        base = [g for r in P if items[r]["fold"] == f for g in dep[r]]
        if mode == "dep":
            return {"type": "FeatureCollection", "features": base}
        if mode == "add":
            return {"type": "FeatureCollection", "features": base + emit(D[(D.fold == f) & (D[src] >= t) & (D.d_cnn > 4.0)], src)}
        return {"type": "FeatureCollection", "features": [g for g in base if g["properties"]["class_name"] not in K] + emit(D[(D.fold == f) & (D[src] >= t)], src)}
    srcs = ["q_tda+scat", "q_all+cnn"]
    D = D.rename(columns={"q_tda+scat": "q_tdascat", "q_all+cnn": "q_allcnn"}); srcs = ["q_tdascat", "q_allcnn"]
    S = [("dep", "", 0)] + [("add", s, t) for s in srcs for t in (0.5, 0.7, 0.9)] + [("rep", s, t) for s in srcs for t in (0.1, 0.2, 0.3)]
    C = fold_counts(make, S, folds, gt, CLS)
    for key in [("dep", "")] + [(m, s) for m in ("add", "rep") for s in srcs]:
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[:2] == key})
        print(f"{key[0]:3s} {key[1]:10s} honest {h:.4f} (oracle {o:.4f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
