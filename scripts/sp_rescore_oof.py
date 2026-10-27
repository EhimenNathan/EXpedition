"""Satellite context rescoring: a second-stage GBM over the fused candidates (D-FINE, RT-DETRv2, heatmap, local
D-FINE no-DN), using features only computable at inference:
  member scores, fused score, class, scene geometry (distance to the nearest scene centre / scene half-size, small or
  big scene, hillfort-scene p), neighbourhood support (candidates and same-family candidates within 15/40/100/300 m,
  their max / mean score), region context (number of candidates, score rank).
Label: candidate point inside a GT polygon of the same class. Trained on the other folds only (nested); per-class
thresholds chosen on the other folds. The hillfort gate is applied afterwards as deployed.

  python scripts/sp_rescore_oof.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds  # noqa: E402
from sp_fusion_oof import fc  # noqa: E402
from sp_fusion4_oof import gate  # noqa: E402

K = ("kurgany_povrezhdennye", "kurgany_tselye")
fam = lambda c: "k" if c in K else c
CLS = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
NAMES = ["dfine", "rtdetr", "hm", "dfnodn"]


def fuse_members(models, match_m=8.0):
    """koz3.sp_fuse.fuse, keeping every member's best score (features)."""
    allc = pd.concat([m.assign(mi=i) for i, m in enumerate(models)], ignore_index=True)
    allc["fam"] = allc.cls.map(fam)
    out = []
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
            out.append((reg, g.cls[i], float(best.mean()), g.x[i], g.y[i], *best))
    return pd.DataFrame(out, columns=["region", "cls", "score", "x", "y"] + [f"s_{n}" for n in NAMES[:len(models)]])


def scene_geometry():
    """region → list of (cx, cy, half) in EPSG:3857 (half = half of the longer side, metres)."""
    from pyproj import Transformer
    from rasterio.transform import Affine
    S = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/work/sp/scenes.csv"))
    crs = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site")
    out = {}
    for r in S.itertuples():
        T = Affine(*json.loads(r.T)); c = f"EPSG:{int(crs.loc[r.site, 'json_epsg'])}" if r.site in crs.index else r.crs
        tr = Transformer.from_crs(c, 3857, always_xy=True)
        (x0, y0), (x1, y1) = tr.transform(*(T * (0, 0))), tr.transform(*(T * (r.W, r.H)))
        out.setdefault(r.site, []).append(((x0 + x1) / 2, (y0 + y1) / 2, max(abs(x1 - x0), abs(y1 - y0)) / 2, r.W * r.H < 4_000_000 / 4))
    return out


def features(F, geom, S):
    F = F.copy()
    F["cls_i"] = F.cls.map(CLS.index); F["kfam"] = F.cls.isin(K).astype(int)
    rel, half, small = [], [], []
    for r in F.itertuples():
        g = geom.get(r.region, [(r.x, r.y, 1e4, False)])
        d = [(np.hypot(r.x - cx, r.y - cy) / max(h, 1), h, sm) for cx, cy, h, sm in g]
        b = min(d); rel.append(b[0]); half.append(b[1]); small.append(int(b[2]))
    F["d_rel"], F["half"], F["small"] = rel, half, small
    p = S.groupby("region").p.max(); F["p_gor"] = F.region.map(p).fillna(-1)
    for c in ("n15", "n40", "n100", "n300", "k40", "k100", "k300", "mx40", "mx100", "mean100", "rank", "n_reg", "max_reg"):
        F[c] = 0.0
    for reg, g in F.groupby("region"):
        xy = g[["x", "y"]].values; t = cKDTree(xy); s = g.score.values; kf = g.kfam.values
        for R, nm in ((15, "n15"), (40, "n40"), (100, "n100"), (300, "n300")):
            nb = t.query_ball_point(xy, R)
            F.loc[g.index, nm] = [len(n) - 1 for n in nb]
            if R in (40, 100, 300):
                F.loc[g.index, f"k{R}"] = [sum(kf[j] for j in n) - kf[i] for i, n in enumerate(nb)]
            if R in (40, 100):
                F.loc[g.index, f"mx{R}"] = [max([s[j] for j in n if j != i], default=0) for i, n in enumerate(nb)]
            if R == 100:
                F.loc[g.index, "mean100"] = [np.mean([s[j] for j in n if j != i]) if len(n) > 1 else 0 for i, n in enumerate(nb)]
        F.loc[g.index, "rank"] = pd.Series(s, index=g.index).rank(ascending=False) / len(g)
        F.loc[g.index, "n_reg"] = len(g); F.loc[g.index, "max_reg"] = s.max()
    return F


def label(F, L):
    from shapely import STRtree
    from shapely.geometry import Point
    y = np.zeros(len(F), int)
    for c, g in L.groupby("class_name"):
        geoms = list(g.geometry); tree = STRtree(geoms)
        idx = np.where(F.cls.values == c)[0]
        best = {}                                   # GT polygon → its highest-scoring candidate (others = duplicates = FP)
        for i in idx:
            pt = Point(F.x.values[i], F.y.values[i])
            for j in tree.query(pt):
                if geoms[j].contains(pt) and F.score.values[i] > best.get(j, (-1, -1))[0]:
                    best[j] = (F.score.values[i], i)
        for _, i in best.values():
            y[i] = 1
    return y


FEATS = None


def main(min_score=0.1):
    import geopandas as gpd
    import lightgbm as lgb
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    M = models(extra=["dfnodn"])
    regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    cache = os.path.join(ROOT, "work/spor/rescore_feats.parquet")
    if os.path.exists(cache):
        F = pd.read_parquet(cache)
    else:
        F = fuse_members([M[n] for n in NAMES])
        F = F[F.score >= min_score * 0.5].reset_index(drop=True)
        F = features(F, scene_geometry(), S)
        F["lab"] = label(F, L); F["fold"] = F.region.map(RF)
        F.to_parquet(cache)
    print(len(F), "candidates;", int(F.lab.sum()), "inside a same-class label")
    feats = [c for c in F.columns if c not in ("region", "cls", "x", "y", "fold", "lab", "p")]
    from rescore_models import oof_predict
    kinds = os.environ.get("SP_RESCORE_MODELS", "lgbm").split(",")       # e.g. lgbm,catboost,ngboost,avg,catboost+anom
    for kd in kinds:
        F["p_" + kd.replace("+", "_")] = oof_predict(F, feats, kd)
    F["p"] = F["p_" + kinds[0].replace("+", "_")]
    from sklearn.metrics import roc_auc_score
    for c in CLS:
        g = F[F.cls == c]
        if g.lab.nunique() > 1:
            print(f"  {c:24s} AUC fused {roc_auc_score(g.lab, g.score):.3f} → " + " ".join(f"{kd} {roc_auc_score(g.lab, g['p_' + kd.replace('+', '_')]):.3f}" for kd in kinds)
                  + f"  (n={len(g)}, pos={int(g.lab.sum())})")
    F.to_parquet(os.path.join(ROOT, "work/spor/rescore_oof.parquet"))
    TH = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)
    for kd in kinds:
        F["b_" + kd.replace("+", "_")] = np.sqrt(F.score * F["p_" + kd.replace("+", "_")])
    srcs = ("score",) + tuple(f"{pre}_{kd.replace('+', '_')}" for kd in kinds for pre in ("p", "b"))

    def make(st, f):
        src, t = st
        G = F[(F.fold == f) & (F[src] >= t)].assign(score=lambda d: d[src])
        return fc(gate(G[["region", "cls", "score", "x", "y"]], S[S.region.map(RF) == f]))
    C = fold_counts(make, [(s_, t) for s_ in srcs for t in TH], folds, gt)
    from ae_det_oof import f1
    from koz3.score import WEIGHTS
    for allowed in tuple((s_,) for s_ in srcs):          # per class, nested: setting chosen on the other folds
        tot = {}
        for c in CLS:
            for f in folds:
                cand = [(s_, t) for s_ in allowed for t in TH]
                oth = {k: np.sum([C[k][g].get(c, (0, 0, 0)) for g in folds if g != f], 0) for k in cand}
                k = max(cand, key=lambda k: f1(*oth[k]))
                tot[c] = np.add(tot.get(c, (0, 0, 0)), C[k][f].get(c, (0, 0, 0)))
        pres = [c for c in CLS if tot[c][0] + tot[c][2]]
        h = sum(WEIGHTS[c] * f1(*tot[c]) for c in pres) / sum(WEIGHTS[c] for c in pres)
        print(f"{'+'.join(allowed):16s} per-class nested {h:.4f}  " + " ".join(f"{c[:8]} {f1(*tot[c]):.3f}" for c in pres))

if __name__ == "__main__":
    main()
