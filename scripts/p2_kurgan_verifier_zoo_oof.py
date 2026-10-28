"""Phase 2 · E2: satellite kurgan verifier zoo in the EXACT deployed v21 / v22 satellite decoding, nested + robust.

Candidates: deployed fusion of the six members (sp_dual_oof.fuse_cls), score ≥ 0.10. Label: the candidate is the best
(highest-scoring) kurgan candidate inside a kurgan label (duplicates near a true object count as negatives, as the metric).
Feature groups per candidate:
  det   fused score, damaged / intact member maxima, their ratio
  ctx   distance to the nearest scene centre / half-size, small scene, scene hillfort p, neighbour counts and maxima
  img   image processing on a 64 m chip at 0.5 m: multi-scale LoG (3/5/8/12 m), DoG, centre-ring contrast, radial
        asymmetry, local std, gradient energy, Laplacian variance
  emb   DINOv3-SAT chip embedding (CLS + mean patch; cache work/spor/oof_dinov3.npz from sp_dinov3_verifier_oof.py),
        PCA 32 fitted on the training folds only
Models (out of fold over the deployed folds): lgbm, catboost, xgboost, ngboost, adaboost, extratrees, logistic,
optuna-tuned lgbm (inner group CV on the training folds), and 'zoo' = rank mean of all.
Then each model's probability v is blended into the kurgan scores, score' = (1 − λ)·score + λ·v (λ = 0 → deployed), the
deployed decoding is re-run, and (model, λ) is chosen on the other folds.
  python scripts/p2_kurgan_verifier_zoo_oof.py [--quick]
Accept only if nested honest > deployed, no outer fold regresses, k* ≥ 3, bootstrap P(gain>0) ≥ 0.8.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "experiments", "claude"))
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
LAMBDAS = (0.0, 0.25, 0.5, 0.75)
NJ = os.cpu_count()


def chip_features(c):
    """64 m chip at 0.5 m (uint8 HWC 128 × 128) → image-processing features at the centre."""
    from scipy import ndimage as ndi
    g = c.astype(np.float32).mean(2); g = (g - g.mean()) / (g.std() + 1e-6)
    H, W = g.shape; cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]; r = np.hypot(yy - cy, xx - cx) * 0.5             # metres
    f = {}
    for rm in (3, 5, 8, 12):
        s = rm / 0.5 / np.sqrt(2); L = -ndi.gaussian_laplace(g, s) * s * s
        f[f"log{rm}"] = float(L[cy - 2:cy + 3, cx - 2:cx + 3].mean()); f[f"alog{rm}"] = abs(f[f"log{rm}"])
    f["dog"] = float((ndi.gaussian_filter(g, 2) - ndi.gaussian_filter(g, 6))[cy - 2:cy + 3, cx - 2:cx + 3].mean())
    cen, ring = g[r < 5].mean(), g[(r >= 8) & (r < 15)].mean()
    f["cen_ring"] = float(cen - ring); f["cen"] = float(cen)
    ang = np.arctan2(yy - cy, xx - cx); sect = [g[(r >= 5) & (r < 12) & (ang >= a) & (ang < a + np.pi / 4)].mean() for a in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
    f["asym"] = float(np.std(sect))
    f["std_loc"] = float(g[r < 12].std())
    gx, gy = ndi.sobel(g, 1), ndi.sobel(g, 0); f["grad"] = float(np.hypot(gx, gy)[r < 12].mean())
    f["lapvar"] = float(ndi.laplace(g)[r < 12].var())
    return f


def _region_feats(region, idx, xs, ys, crs_tab):
    """Windowed reads (64 m around each candidate, resampled to 0.5 m = 128 px): memory stays small even for 300 Mpx
    scenes. Same chip geometry as sp_dinov3_probe.chip."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from pyproj import Transformer
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data/train-expds-3", region, "04_*", "*.tif"))):
        with rasterio.open(path) as ds:
            crs = f"EPSG:{int(crs_tab.loc[region, 'json_epsg'])}" if region in crs_tab.index else ds.crs.to_string()
            X, Y = Transformer.from_crs(3857, crs, always_xy=True).transform(xs, ys)
            T = ds.transform; res = abs(T.a); half = 32.0 / res; inv = ~T; nb = min(3, ds.count)
            for j in range(len(idx)):
                if idx[j] in out:
                    continue
                c, r = inv * (X[j], Y[j])
                r0, c0, r1, c1 = int(r - half), int(c - half), int(r + half), int(c + half)
                if r0 < 0 or c0 < 0 or r1 > ds.height or c1 > ds.width or r1 <= r0 or c1 <= c0:
                    continue
                a = ds.read(indexes=list(range(1, nb + 1)), window=Window(c0, r0, c1 - c0, r1 - r0), out_shape=(nb, 128, 128),
                            resampling=Resampling.bilinear).astype(np.float32)
                if nb == 1:
                    a = np.repeat(a, 3, 0)
                out[idx[j]] = chip_features(np.clip(a, 0, 255).transpose(1, 2, 0))
    return out


def image_features(F, cache):
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    if os.path.exists(cache):
        D = pd.read_parquet(cache)
        if len(D) == len(F):
            return D
    crs_tab = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site")
    groups = list(F.groupby("region").groups.items())
    res = Parallel(n_jobs=min(NJ, 16))(delayed(_region_feats)(r, np.asarray(list(ix)), F.x.values[list(ix)], F.y.values[list(ix)], crs_tab)
                              for r, ix in tqdm(groups, desc="image features (regions)"))
    allf = {k: v for d in res for k, v in d.items()}
    cols = list(next(iter(allf.values())).keys())
    D = pd.DataFrame([allf.get(i, {c: np.nan for c in cols}) for i in range(len(F))])
    D.to_parquet(cache)
    return D


def context_features(F, Sg, geom):
    from scipy.spatial import cKDTree
    out = np.zeros((len(F), 9))
    p_reg = Sg.groupby("region").p.max()
    for reg, g in F.groupby("region"):
        ix = g.index.values; xy = g[["x", "y"]].values; t = cKDTree(xy); s = g.score.values
        kf = g.cls.isin(K).values
        sc = geom.get(reg, [])
        for n, i in enumerate(ix):
            d = min([(np.hypot(xy[n, 0] - cx, xy[n, 1] - cy) / max(h, 1), h, sm) for cx, cy, h, sm in sc] or [(0.0, 1e4, False)])
            out[i, 0], out[i, 1], out[i, 2] = d[0], np.log1p(d[1]), float(d[2])
        for k_, R in enumerate((30.0, 100.0)):
            nb = t.query_ball_point(xy, R)
            out[ix, 3 + 2 * k_] = [sum(kf[j] for j in q) - kf[n] for n, q in enumerate(nb)]
            out[ix, 4 + 2 * k_] = [max([s[j] for j in q if j != n], default=0) for n, q in enumerate(nb)]
        out[ix, 7] = p_reg.get(reg, -1); out[ix, 8] = len(g)
    return pd.DataFrame(out, columns=["d_rel", "log_half", "small", "nk30", "mx30", "nk100", "mx100", "p_gor", "n_reg"])


def label_best(F, L):
    """1 for the highest-scoring kurgan candidate inside each kurgan label (others inside = duplicates = 0)."""
    from shapely import STRtree
    from shapely.geometry import Point
    y = np.zeros(len(F), int); kl = L[L.class_name.isin(K)]
    for reg, g in F[F.cls.isin(K)].groupby("region"):
        lab = kl[kl.region == reg]
        if not len(lab):
            continue
        geoms = list(lab.geometry); tree = STRtree(geoms); best = {}
        for i, x, yy, s in zip(g.index, g.x, g.y, g.score):
            p = Point(x, yy)
            for j in tree.query(p):
                if geoms[j].contains(p) and s > best.get(j, (-1, -1))[0]:
                    best[j] = (s, i)
        for _, i in best.values():
            y[i] = 1
    return y


def model_zoo(quick=False):
    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    Z = {
        "lgbm": lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                                           subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, n_jobs=NJ, verbose=-1, random_state=0),
        "catboost": lambda: CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=5, verbose=0, random_seed=0,
                                               thread_count=NJ, allow_writing_files=False),
        "xgboost": lambda: xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                                             tree_method="hist", n_jobs=NJ, random_state=0, eval_metric="logloss"),
        "adaboost": lambda: AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=0),
        "extratrees": lambda: ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, max_features=0.5, n_jobs=NJ, random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
    }
    if not quick:
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        from sklearn.tree import DecisionTreeRegressor
        Z["ngboost"] = lambda: NGBClassifier(Dist=Bernoulli, n_estimators=400, learning_rate=0.03, verbose=False, random_state=0,
                                             Base=DecisionTreeRegressor(max_depth=4, min_samples_leaf=40))
    return Z


def optuna_lgbm(X, y, groups, n_trials):
    """LightGBM tuned by Optuna on GROUP (region) 3-fold CV inside the training folds only → best estimator."""
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        prm = dict(n_estimators=t.suggest_int("n", 200, 1200), learning_rate=t.suggest_float("lr", 0.01, 0.1, log=True),
                   num_leaves=t.suggest_int("leaves", 7, 63), min_child_samples=t.suggest_int("mcs", 10, 120),
                   colsample_bytree=t.suggest_float("col", 0.4, 1.0), subsample=t.suggest_float("sub", 0.5, 1.0), subsample_freq=1,
                   reg_lambda=t.suggest_float("l2", 1e-3, 10, log=True), n_jobs=NJ, verbose=-1, random_state=0)
        ap = []
        for tr, va in GroupKFold(3).split(X, y, groups):
            m = lgb.LGBMClassifier(**prm).fit(X[tr], y[tr]); ap.append(average_precision_score(y[va], m.predict_proba(X[va])[:, 1]))
        return float(np.mean(ap))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    b = st.best_params
    return lgb.LGBMClassifier(n_estimators=b["n"], learning_rate=b["lr"], num_leaves=b["leaves"], min_child_samples=b["mcs"],
                              colsample_bytree=b["col"], subsample=b["sub"], subsample_freq=1, reg_lambda=b["l2"], n_jobs=NJ,
                              verbose=-1, random_state=0)


def main(quick=False):
    import geopandas as gpd
    from scipy.stats import rankdata
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, average_precision_score
    from tqdm.auto import tqdm
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls
    from sp_dinov3_verifier_oof import embeddings
    from robust_accept import site_counts, report
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    F0 = F0[F0.score >= 0.10].reset_index(drop=True)
    kur = F0.cls.isin(K).values; fold = F0.region.map(RF).values
    y = label_best(F0, L)
    print(f"{len(regions)} regions | {len(F0)} candidates (score ≥ 0.10), kurgan {int(kur.sum())}, positives {int(y.sum())}", flush=True)
    E, ok = embeddings(F0, os.path.join(ROOT, "work/spor/oof_dinov3.npz"))
    IMG = image_features(F0, os.path.join(ROOT, "work/spor/p2_img_feats.parquet"))
    CTX = context_features(F0, Sg, geom)
    DET = pd.DataFrame({"score": F0.score, "s_kp": F0.s_kp, "s_kt": F0.s_kt, "ratio": F0.s_kp / (F0.s_kp + F0.s_kt + 1e-6)})
    base = pd.concat([DET, CTX, IMG], axis=1).to_numpy(np.float32)
    zoo = model_zoo(quick); V = {k: np.full(len(F0), np.nan) for k in list(zoo) + ["optuna_lgbm"]}
    for f in tqdm(folds, desc="outer folds"):
        tr = (fold != f) & kur; te = (fold == f) & kur
        pca = PCA(32, random_state=0).fit(E[tr & ok]); emb = pca.transform(E) * ok[:, None]
        X = np.nan_to_num(np.c_[base, emb])
        for name, mk in tqdm(list(zoo.items()), desc=f"fold {f} models", leave=False):
            V[name][te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        V["optuna_lgbm"][te] = optuna_lgbm(X[tr], y[tr], F0.region.values[tr], 10 if quick else 40).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    V["zoo"] = np.full(len(F0), np.nan)
    V["zoo"][kur] = np.mean([rankdata(V[k][kur]) / kur.sum() for k in V if k != "zoo"], 0)
    print("kurgan candidates, out of fold:  " + f"det score AUC {roc_auc_score(y[kur], F0.score[kur]):.3f} AP {average_precision_score(y[kur], F0.score[kur]):.3f}")
    for k in V:
        print(f"   {k:12s} AUC {roc_auc_score(y[kur], V[k][kur]):.3f}  AP {average_precision_score(y[kur], V[k][kur]):.3f}", flush=True)
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0

    def pipeline(st, regs):
        name, lam = st
        F = F0.copy()
        if lam > 0:
            F.loc[kur, "score"] = (1 - lam) * F.score[kur] + lam * V[name][kur]
        F = F[F.region.isin(regs)]
        F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)]; keep = []
        for r in F.itertuples():
            d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
            keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
        F = F[np.array(keep, bool)]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
        G = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        Sx = Sg[Sg.region.isin(regs)]
        G = G[~((G.cls == "gorodishcha") & G.region.isin(set(Sx[Sx.p < 0.1].region)))]
        rows = []
        for r in Sx[Sx.p >= 0.1].itertuples():
            cx, cy = r.x, r.y3857
            c = GC[(GC.region == r.region) & (np.hypot(GC.x - cx, GC.y - cy) <= 0.2 * half[r.Index])]
            if len(c):
                b = c.loc[c.score.idxmax()]; cx, cy = b.x, b.y
            rows.append((r.region, "gorodishcha", r.p, cx, cy))
        return pd.concat([G, pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)

    S = [("none", 0.0)] + [(k, lam) for k in V for lam in LAMBDAS[1:]]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    C = {}
    for st in tqdm(S, desc="exact v21/v22 decoding per setting"):
        C.update(fold_counts(lambda s, f: fc(pipeline(s, [r for r in regions if RF[r] == f])), [st], folds, gt))
    for st in S:
        tot = _add_all(C[st].values())
        print(f"{str(st):26s} pooled {weighted(tot):.4f}  folds [{' '.join(f'{weighted(C[st][f]):.4f}' for f in folds)}]", flush=True)
    h, ch, o, os_, _ = nested(C); b0 = weighted(_add_all(C[("none", 0.0)].values()))
    print(f"deployed {b0:.4f} | nested honest (model and λ chosen) {h:.4f} ({h - b0:+.4f}) | best fixed {o:.4f} @ {os_} | chosen {ch}")
    print(f"worst outer-fold change {min(weighted(C[ch[f]][f]) - weighted(C[('none', 0.0)][f]) for f in folds):+.4f}")
    for k in V:                                       # per-model nested (λ only), to see which family carries signal
        Ck = {s: C[s] for s in S if s[0] in ("none", k)}
        hk, chk, *_ = nested(Ck); print(f"   {k:12s} λ-only nested {hk:.4f} ({hk - b0:+.4f}) chosen {list(chk.values())}")
    st = max(set(ch.values()), key=list(ch.values()).count)
    if st != ("none", 0.0):
        gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
        CS = site_counts(lambda s, r: fc(pipeline(s, [r])), (("none", 0.0), st), regions, gtr)
        report(f"kurgan verifier {st}", CS[("none", 0.0)], CS[st])


if __name__ == "__main__":
    main("--quick" in sys.argv)
