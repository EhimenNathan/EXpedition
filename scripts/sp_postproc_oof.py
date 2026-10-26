"""Satellite post-processing tests on the v12 pipeline (5-member fusion, global 0.25 / architecture 0.15, hillfort gate),
all nested over the 4 site folds:
  pos   — fused point = score-weighted mean of the members' positions (WBF-style) instead of the top member's
  rank  — each member's scores rank-normalised (per class family, on its own candidates) before fusion
  stack — logistic stacking of the 5 member scores (+ #members), trained on the other folds ("best candidate per GT")
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds  # noqa: E402
from sp_fusion_oof import fc  # noqa: E402
from sp_fusion4_oof import gate  # noqa: E402

K = ("kurgany_povrezhdennye", "kurgany_tselye")
fam = lambda c: "k" if c in K else c
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq")


def fuse_full(models, match_m=8.0):
    """koz3.sp_fuse.fuse + member score vector + score-weighted mean position."""
    allc = pd.concat([m.assign(mi=i) for i, m in enumerate(models)], ignore_index=True)
    allc["fam"] = allc.cls.map(fam); out = []
    for (reg, f), g in allc.groupby(["region", "fam"]):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        tree = cKDTree(g[["x", "y"]].values)
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            idx = [j for j in tree.query_ball_point(g.loc[i, ["x", "y"]].values, match_m * k) if not used[j]]
            used[idx] = True
            best = np.zeros(len(models))
            for j in idx:
                best[g.mi[j]] = max(best[g.mi[j]], g.score[j])
            w = g.score.values[idx]
            out.append((reg, g.cls[i], float(best.mean()), g.x[i], g.y[i], float(np.average(g.x.values[idx], weights=w)),
                        float(np.average(g.y.values[idx], weights=w)), int((best > 0).sum()), *best))
    return pd.DataFrame(out, columns=["region", "cls", "score", "x", "y", "xw", "yw", "n_m"] + [f"s{i}" for i in range(len(models))])


def label_best(F, L):
    from shapely import STRtree
    from shapely.geometry import Point
    y = np.zeros(len(F), int)
    for c, g in L.groupby("class_name"):
        geoms = list(g.geometry); tree = STRtree(geoms); best = {}
        for i in np.where(F.cls.values == c)[0]:
            pt = Point(F.x.values[i], F.y.values[i])
            for j in tree.query(pt):
                if geoms[j].contains(pt) and F.score.values[i] > best.get(j, (-1, -1))[0]:
                    best[j] = (F.score.values[i], i)
        for _, i in best.values():
            y[i] = 1
    return y


def main():
    import geopandas as gpd
    from sklearn.linear_model import LogisticRegression
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    M = models(extra=["dfnodn", "balq"]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    F = fuse_full([M[n] for n in NAMES]); F["fold"] = F.region.map(RF)
    # rank-normalised members (each member's score → its rank percentile within class family)
    R = []
    for n in NAMES:
        m = M[n].copy(); m["score"] = m.groupby(m.cls.map(fam)).score.rank(pct=True) * m.score.max()
        R.append(m)
    FR = fuse_full(R); FR["fold"] = FR.region.map(RF)
    # stacking
    F["lab"] = label_best(F, L)
    X = F[[f"s{i}" for i in range(5)] + ["n_m"]].values; F["p_stack"] = np.nan
    for f in folds:
        tr = F.fold != f
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], F.lab[tr])
        F.loc[~tr, "p_stack"] = clf.predict_proba(X[~tr])[:, 1]
    # map stacked probability onto the fused-score scale per fold (monotone, same candidate count above 0.25)
    def emit(D, src, t, t_arch, pos):
        G = D[D[src] >= np.where(D.cls == "arkhitektury", t_arch, t)].copy()
        G["score"] = G[src]
        if pos:
            G["x"], G["y"] = G.xw, G.yw
        return G[["region", "cls", "score", "x", "y"]]

    def make(s, f):
        mode, t = s
        if mode == "base":
            G = emit(F[F.fold == f], "score", t, 0.15, False)
        elif mode == "pos":
            G = emit(F[F.fold == f], "score", t, 0.15, True)
        elif mode == "rank":
            G = emit(FR[FR.fold == f], "score", t, 0.15 * t / 0.25, False)
        else:
            G = emit(F[F.fold == f], "p_stack", t, t * 0.6, False)
        return fc(gate(G, S[S.region.map(RF) == f]))
    SS = [("base", 0.25), ("pos", 0.25)] + [("rank", t) for t in (0.4, 0.5, 0.6, 0.7, 0.8)] + [("stack", t) for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)]
    C = fold_counts(make, SS, folds, gt)
    for mode in ("base", "pos", "rank", "stack"):
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[0] == mode})
        print(f"{mode:6s} honest {h:.4f} (oracle {o:.4f} @ {os_[1]}) chosen {[c[1] for c in ch.values()]}  "
              + " ".join(f"{c[:6]} {2 * v[0] / max(2 * v[0] + v[2] + 0.5 * v[1], 1):.3f}" for c, v in tot.items() if v[0] + v[2]))


if __name__ == "__main__":
    main()
