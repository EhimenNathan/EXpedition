"""Site-agnostic classical mound detector as a recall source where the CNN is blind, honest over the site folds.

Data (scripts/li_relief_signal.py): 63 % of CNN-missed kurgans have ≥ 0.3 m relief prominence, but background bumps
≥ 0.3 m number ~5 200/km² → prominence alone is useless; shape / scale / context must separate them.
Candidates: local maxima (6 m window) of the 10 m detrended DTM smoothed with a 1 m Gaussian, ≥ 0.15 m.
Features (robust, mostly scale ratios / site-normalised): multi-scale detrended relief at the peak (2/5/10/25 m), disk
means of the smoothed relief (3/6/10 m), spikiness r2/r10, slope and Laplacian, radial symmetry (8 directions at 3 and
5 m: mean, std, min → circularity), site ranks of r10 / r5, site candidate density, neighbour counts (25/100 m) of
strong candidates, CNN blend heat / masks at the peak. Label: peak inside a kurgan polygon.
LightGBM per fold on the other folds. Integration (nested): candidates with q ≥ τ and no deployed kurgan within 6 m are
added (class from the masks, dual 0.05) to the deployed decode (blend + prior, thr 0.15).
  python scripts/li_blob_gbm_oof.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def site_candidates(x, valid, heat, mk, res):
    from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter
    r2, r5, r10, r25 = (np.asarray(x[i], np.float32) * 0.5 for i in range(4))
    slope, lap = np.asarray(x[4], np.float32), np.asarray(x[7], np.float32)
    s10 = gaussian_filter(r10, 1.0 / res)
    v = np.asarray(valid, bool)
    pk = (s10 == maximum_filter(s10, size=int(round(6 / res)))) & v & (s10 >= 0.15)
    ys, xs = np.nonzero(pk)
    if not len(ys):
        return None
    H, W = s10.shape
    F = {"y": ys, "x": xs}
    for nm, a in (("r2", r2), ("r5", r5), ("r10", r10), ("r25", r25), ("s10", s10), ("slope", slope), ("lap", lap)):
        F[nm] = a[ys, xs]
    for d in (3, 6, 10):
        F[f"m{d}"] = uniform_filter(s10, size=int(round(2 * d / res)))[ys, xs]
    F["slope6"] = uniform_filter(slope, size=int(round(12 / res)))[ys, xs]
    for rad in (3, 5):
        ring = []
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            yy = np.clip((ys + rad / res * np.sin(a)).astype(int), 0, H - 1); xx = np.clip((xs + rad / res * np.cos(a)).astype(int), 0, W - 1)
            ring.append(s10[yy, xx])
        ring = np.stack(ring)
        drop = F["s10"][None] - ring
        F[f"drop{rad}_mean"], F[f"drop{rad}_std"], F[f"drop{rad}_min"] = drop.mean(0), drop.std(0), drop.min(0)
        F[f"circ{rad}"] = drop.min(0) / (drop.mean(0) + 1e-3)
    F["spiky"] = F["r2"] / (np.abs(F["r10"]) + 0.05)
    F["shape"] = F["m3"] / (F["s10"] + 0.05)
    F["heat"] = heat[ys, xs]; F["heat_max5"] = maximum_filter(heat, size=int(round(10 / res)))[ys, xs]
    F["p_dam"], F["p_int"] = mk[0][ys, xs], mk[1][ys, xs]
    D = pd.DataFrame(F)
    for c in ("r10", "r5", "s10", "drop5_mean"):
        D[f"rank_{c}"] = D[c].rank(pct=True)
    D["site_dens"] = len(D) / max(v.sum() * res * res / 1e6, 1e-3)
    D["site_r10_med"] = D.r10.median()
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[xs, ys]); strong = D[(D.drop5_mean > 0.2) & (D.circ5 > 0.3)]
    ts = cKDTree(strong[["x", "y"]].values) if len(strong) else None
    for R in (25, 100):
        D[f"n{R}"] = [len(q) - 1 for q in t.query_ball_point(np.c_[xs, ys], R / res)]
        D[f"ns{R}"] = [len(q) for q in ts.query_ball_point(np.c_[xs, ys], R / res)] if ts is not None else 0
    return D


def main():
    import lightgbm as lgb
    from pyproj import Transformer
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree
    from sklearn.metrics import roc_auc_score
    from koz3.li_train import labels, site_items, decode_site_v2, CLS, N_HM
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from robust_accept import site_counts, report
    from val_sim import pts
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    sites = sorted(P["ens"]); fold = {r: items[r]["fold"] for r in sites}; folds = sorted(set(fold.values()))
    pri = {"R": 200.0, "K": 3, "snb": 0.2}; base, cands = {}, []
    for r in sites:
        it = items[r]; meta = it["meta"]; res = meta["res"]
        b = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32)
        base[r] = decode_site_v2(b, it["valid"], dict(meta, region=r), thr=0.15, dual=0.05, comp_thr=0.5, prior=pri)
        D = site_candidates(it["x"], it["valid"], b[0], b[N_HM:N_HM + 2], res)
        if D is None:
            continue
        polys = [g for c, g in it["objs"] if c in (0, 1)]; tree = STRtree(polys) if polys else None
        D["lab"] = [int(tree is not None and any(polys[j].contains(Point(px, py)) for j in tree.query(Point(px, py)))) for px, py in zip(D.x, D.y)]
        tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True)
        D["X"], D["Y"] = tr.transform(meta["x0"] + D.x.values * res, meta["ytop"] - D.y.values * res)
        bk = [shape(f["geometry"]).centroid for f in base[r] if f["properties"]["class_name"] in K]
        if bk:
            from scipy.spatial import cKDTree
            kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(D.Y.mean() / 6378137.0)) - np.pi / 2)))
            D["covered"] = cKDTree(np.array([(p.x, p.y) for p in bk])).query(np.c_[D.X, D.Y])[0] <= 6 * kk
        else:
            D["covered"] = False
        D["region"], D["fold"] = r, fold[r]
        cands.append(D); print(r, len(D), int(D.lab.sum()), "covered", int(D.covered.sum()), flush=True)
    C_ = pd.concat(cands, ignore_index=True)
    feats = [c for c in C_.columns if c not in ("y", "x", "X", "Y", "lab", "region", "fold", "covered")]
    C_["q"] = np.nan
    for f in folds:
        trn = C_[C_.fold != f]
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=16).fit(trn[feats], trn.lab)
        C_.loc[C_.fold == f, "q"] = m.predict_proba(C_.loc[C_.fold == f, feats])[:, 1]
    C_.to_parquet("work/li_blob_cands.parquet")
    print(f"{len(C_)} candidates, {int(C_.lab.sum())} inside kurgans; OOF AUC {roc_auc_score(C_.lab, C_.q):.3f}"
          f" (heat only {roc_auc_score(C_.lab, C_.heat):.3f}); uncovered: {int((~C_.covered).sum())} cands, {int(C_.lab[~C_.covered].sum())} kurgan hits")
    U = C_[~C_.covered]
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        s = U[U.q >= t]; print(f"  uncovered q ≥ {t}: {len(s)} cands, hit rate {s.lab.mean() if len(s) else 0:.3f}, kurgan hits {int(s.lab.sum())}")
    imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False); print("top features:", ", ".join(imp.index[:12]))

    def site_fc(st, r):
        fs = list(base[r])
        if st[0] != "base":
            tau = st[1]; g = U[(U.region == r) & (U.q >= tau)]
            if len(g):
                cls = np.where(g.p_dam >= g.p_int, K[0], K[1]); oth = np.where(cls == K[0], K[1], K[0])
                tot = g.p_dam + g.p_int + 1e-6; po = np.where(cls == K[0], g.p_int, g.p_dam) / tot
                rows = pd.DataFrame({"region": r, "cls": cls, "score": g.q.values * 0.5, "x": g.X.values, "y": g.Y.values})
                dual = pd.DataFrame({"region": r, "cls": oth, "score": g.q.values * 0.49, "x": g.X.values, "y": g.Y.values})[po >= 0.05]
                fs = fs + pts(pd.concat([rows, dual]), "li")
        return {"type": "FeatureCollection", "features": fs}
    SS = [("base",)] + [("add", t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    C = fold_counts(lambda st, f: {"type": "FeatureCollection", "features": [g for r in sites if fold[r] == f for g in site_fc(st, r)["features"]]}, SS, folds, gt, CLS)
    for st in SS:
        tot = _add_all(C[st].values())
        print(f"{st}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  kt {tuple(int(v) for v in tot[K[1]])}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    CS = site_counts(site_fc, (SS[0], os_), sites, gtr, CLS)
    report(f"LiDAR classical mound detector {os_}", CS[SS[0]], CS[os_])


if __name__ == "__main__":
    main()
