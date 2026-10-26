"""Learned LiDAR post-processor (rank-to-F1), nested over the site folds, inside the EXACT deployed decoder:
kurgan peaks of the deployed blend ((s0+s1)/2, aug, augx in thirds) get features — blend / per-model peak heights,
agreement, spread, damaged / intact mask shares, neighbour counts (25/100/200/500 m) and strongest neighbour, local relief /
slope, site-level statistics of the site's own predictions — and a LightGBM trained on the other folds predicts "best
match of a kurgan label". That probability replaces the peak height in decode_site_v2 (same class / dual / threshold
logic). Compared with the deployed decoding and the neighbourhood prior (R 200, K 3, snb 0.2).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main():
    import lightgbm as lgb
    import koz3.hm as hm
    import li_compare_oof as lc
    from pyproj import Transformer
    from shapely import STRtree
    from shapely.geometry import Point
    from koz3.li_train import labels, site_items, decode_site_v2, neighbour_prior, CLS, N_HM
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    B = {r: ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32) for r in P["ens"]}
    regions = sorted(B); fold = {r: items[r]["fold"] for r in regions}; folds = sorted(set(fold.values()))
    rows = []
    for r in regions:
        it = items[r]; meta = it["meta"]; b = B[r]; x = np.asarray(it["x"], np.float32)
        pk = [q for q in hm.decode(b[:N_HM], it["valid"], thr=0.05, nms=11) if q[0] == 0]
        if not pk:
            continue
        xy = np.array([(q[1], q[2]) for q in pk]); s = np.array([q[3] for q in pk])
        t = cKDTree(xy); strong = xy[s >= 0.2]; ts = cKDTree(strong) if len(strong) else None
        tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True)
        lab = L[(L.region == r) & L.class_name.isin(K)].to_crs(meta["crs"]); geoms = list(lab.geometry); tree = STRtree(geoms) if geoms else None
        site_n = int((s >= 0.3).sum()); site_med = float(np.median(s)); best = {}
        for i, (c, px, py, sc) in enumerate(pk):
            yi, xi = int(py), int(px); y0, y1, x0, x1 = max(0, yi - 2), yi + 3, max(0, xi - 2), xi + 3
            d = {"region": r, "fold": fold[r], "px": px, "py": py, "s": sc}
            for nm in ("ens", "aug", "augx"):
                d[f"s_{nm}"] = float(P[nm][r][0, yi, xi])
            ms = np.array([d["s_ens"], d["s_aug"], d["s_augx"]]); d["agree"] = int((ms >= 0.1).sum()); d["spread"] = float(ms.std())
            pd_, pi_ = float(b[N_HM][y0:y1, x0:x1].mean()), float(b[N_HM + 1][y0:y1, x0:x1].mean())
            d["p_dam"], d["p_int"], d["ratio"] = pd_, pi_, pd_ / (pd_ + pi_ + 1e-6)
            for R in (25, 100, 200, 500):
                d[f"n{R}"] = len(t.query_ball_point((px, py), R / meta["res"])) - 1
                d[f"ns{R}"] = (len(ts.query_ball_point((px, py), R / meta["res"])) - (sc >= 0.2)) if ts is not None else 0
            nn = t.query((px, py), k=min(6, len(xy)))
            d["nn_max"] = float(max([s[j] for j in np.atleast_1d(nn[1])[1:]], default=0))
            d["lrm10"] = float(x[2, yi, xi]); d["lrm5"] = float(x[1, yi, xi]); d["slope"] = float(x[4, y0:y1, x0:x1].mean())
            d["site_n"] = site_n; d["site_med"] = site_med; d["rank"] = float((s > sc).mean()); d["lab"] = 0
            X_, Y_ = meta["x0"] + px * meta["res"], meta["ytop"] - py * meta["res"]
            if tree is not None:
                pt = Point(X_, Y_)
                for j in tree.query(pt):
                    if geoms[j].contains(pt) and sc > best.get(j, (-1, -1))[0]:
                        best[j] = (sc, len(rows))
            rows.append(d)
        for _, i in best.values():
            rows[i]["lab"] = 1
    D = pd.DataFrame(rows); print(len(D), "kurgan peaks;", int(D.lab.sum()), "best matches")
    feats = [c for c in D.columns if c not in ("region", "fold", "px", "py", "lab")]
    D["q"] = np.nan
    for f in folds:
        trn = D[D.fold != f]
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=16).fit(trn[feats], trn.lab)
        D.loc[D.fold == f, "q"] = m.predict_proba(D.loc[D.fold == f, feats])[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"AUC blend peak {roc_auc_score(D.lab, D.s):.3f} → learned {roc_auc_score(D.lab, D.q):.3f}")
    qmap = {(r_, round(a, 3), round(b_, 3)): q for r_, a, b_, q in zip(D.region, D.px, D.py, D.q)}
    orig = hm.decode; state = {}

    def decode_hook(h, valid=None, thr=0.05, nms=9):
        pk = orig(h, valid, min(thr, 0.05), nms); mode = state["mode"]; out = []
        if mode == "prior":
            pk = neighbour_prior(pk, 200, 3, 0.2, 0.5)
        for c, px, py, sc in pk:
            if c == 0 and mode == "learned":
                sc = float(qmap.get((state["r"], round(px, 3), round(py, 3)), 0.0))
            if sc > thr:
                out.append((c, px, py, sc))
        return out
    hm.decode = decode_hook
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in regions if fold[r] == f])]) for f in folds}
    cache = {}

    def make(st, f):
        mode, t_ = st; state["mode"] = mode; feats_ = []
        for r in regions:
            if fold[r] != f:
                continue
            key = (r, mode, t_)
            if key not in cache:
                state["r"] = r; it = items[r]
                cache[key] = decode_site_v2(B[r], it["valid"], dict(it["meta"], region=r), thr=t_, dual=0.05, comp_thr=0.5)
            feats_ += cache[key]
        return {"type": "FeatureCollection", "features": feats_}
    SS = [("base", 0.15), ("prior", 0.15)] + [("learned", t_) for t_ in (0.03, 0.05, 0.07, 0.1, 0.15)]
    C = fold_counts(make, SS, folds, gt, CLS)
    for st in SS:
        print(f"{st}: pooled {weighted(_add_all(C[st].values())):.4f}", flush=True)
    h, ch, o, os_, _ = nested({k: v for k, v in C.items() if k[0] == "learned"})
    print(f"deployed {weighted(_add_all(C[('base', 0.15)].values())):.4f} | prior {weighted(_add_all(C[('prior', 0.15)].values())):.4f} | learned nested {h:.4f} chosen {list(ch.values())}")


if __name__ == "__main__":
    main()
