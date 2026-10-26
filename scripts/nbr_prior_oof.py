"""Neighbourhood (prediction-density) prior, nested: each candidate's confidence × (0.5 + 0.5·clip(n/K, 0, 2)), n = other
same-class candidates within R (plan constants: kurgans R 200 K 3, hillforts 500/2, fortifications 300/2, architecture
150/2); then the class threshold is applied to the reweighted score (otherwise F1 cannot change). Nested over a
threshold multiplier and prior on/off.
  python scripts/nbr_prior_oof.py sat | li
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
RK = {"kurgany_povrezhdennye": (200, 3), "kurgany_tselye": (200, 3), "gorodishcha": (500, 2), "fortifikatsii": (300, 2), "arkhitektury": (150, 2)}
K2 = ("kurgany_povrezhdennye", "kurgany_tselye")


def weights(D):
    w = np.ones(len(D))
    for (reg, c), g in D.groupby(["region", "cls"]):
        if c not in RK or len(g) < 2:
            continue
        R, Kc = RK[c]; t = cKDTree(g[["x", "y"]].values)
        n = np.array([len(q) - 1 for q in t.query_ball_point(g[["x", "y"]].values, R)])
        w[D.index.get_indexer(g.index)] = 0.5 + 0.5 * np.clip(n / Kc, 0, 2)
    return w


def sat():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_fusion4_oof import gate
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
    M = models(extra=["dfnodn", "balq", "balq3"]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    F = fuse_cls([M[n] for n in names]); F = F[F.score >= 0.1].reset_index(drop=True)
    geom = scene_geometry(); far = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        far.append((not d[1]) and d[0] > 0.5)
    F["far"] = far; F["w"] = weights(F)

    def make(s, f):
        prior, m = s
        G = F[F.region.map(RF) == f].copy()
        sc = G.score * (G.w if prior else 1.0)
        base = np.where(G.cls == "arkhitektury", 0.15, 0.25) * m
        keep = (sc >= base) & (~G.far | (sc >= 0.35 * m))
        G = G[keep].assign(score=sc[keep])
        ex = [(r.region, K2[1] if r.cls == K2[0] else K2[0], r.score * 0.99, r.x, r.y) for r in G[G.cls.isin(K2)].itertuples()
              if (r.s_kt if r.cls == K2[0] else r.s_kp) >= 0.4]
        G = pd.concat([G[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        return fc(gate(G, S[S.region.map(RF) == f]))
    SS = [(False, 1.0)] + [(True, m) for m in (0.8, 1.0, 1.2, 1.4)]
    C = fold_counts(make, SS, folds, gt)
    for s in SS:
        print(f"sat prior {s[0]} thr×{s[1]}: pooled {weighted(_add_all(C[s].values())):.4f}")
    h, ch, o, os_, _ = nested(C)
    print(f"sat v12 {weighted(_add_all(C[(False, 1.0)].values())):.4f} | nested honest {h:.4f} chosen {list(ch.values())}")


def li():
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from shapely.geometry import shape
    sys.path.insert(0, os.path.join(ROOT, "scripts")); import li_compare_oof as lc
    P = lc.load("ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst")
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    fold = {r: items[r]["fold"] for r in P}; folds = sorted(set(fold.values()))
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if fold[r] == f])]) for f in folds}
    feats = {r: decode_site_v2(P[r].astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=0.08, dual=0.05) for r in P}
    rows = [(r, i, f_["properties"]["class_name"], f_["properties"]["confidence"], *shape(f_["geometry"]).centroid.coords[0]) for r, fs in feats.items() for i, f_ in enumerate(fs)]
    D = pd.DataFrame(rows, columns=["region", "i", "cls", "score", "x", "y"]); D["w"] = weights(D)

    def make(s, f):
        prior, t = s; out = []
        for r in P:
            if fold[r] != f:
                continue
            d = D[D.region == r]
            for row in d.itertuples():
                g = feats[r][row.i]
                if row.cls in K2:
                    sc = row.score * (row.w if prior else 1.0)
                    if sc < t:
                        continue
                out.append(g)
        return {"type": "FeatureCollection", "features": out}
    SS = [(False, 0.15)] + [(True, t) for t in (0.1, 0.15, 0.2, 0.25)]
    C = fold_counts(make, SS, folds, gt, CLS)
    for s in SS:
        print(f"li prior {s[0]} thr {s[1]}: pooled {weighted(_add_all(C[s].values())):.4f}")
    h, ch, o, os_, _ = nested(C)
    print(f"li blend {weighted(_add_all(C[(False, 0.15)].values())):.4f} | nested honest {h:.4f} chosen {list(ch.values())}")


if __name__ == "__main__":
    {"sat": sat, "li": li}[sys.argv[1]]()
