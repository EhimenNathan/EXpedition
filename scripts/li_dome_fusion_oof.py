"""LiDAR physics-dome rescue fused into the deployed CNN blend, nested + robust (decisive test for
experiments/claude/li_dome_detector.py).

Evidence (59 sites, leave-site-out, CPU; experiments/claude/li_dome_features.py): physical dome candidates ranked by
a LightGBM on physical features (multi-scale relief, symmetry CV, plan/profile curvature, site-normalised and
neighbourhood features; one positive per mound) reach per-site AUC 0.900 and pooled kurgan F1 0.441 alone; on the
CNN's blindest site 011 Erentuna F1 0.580 alone (CNN 0.341). Here they rescue only where the deployed CNN blend has no
kurgan within `gap` metres.

Steps (no GPU):
1. candidates + features for every LiDAR site (li_dome_features.site; DTM from data/las_prep/<region>.npz → 0.5 m),
   cached in work/li_dome/feat/;
2. ranker trained per LiDAR fold on the other folds (one positive per mound, other in-label candidates excluded);
3. deployed CNN blend decoded exactly as deployed (thr 0.15, dual 0.05, comp 0.5, neighbourhood prior);
4. rescue: add candidates with ranker p ≥ t and no CNN kurgan within gap m, as damaged kurgans (confidence α·p;
   optional intact twin at 0.99·α·p, as the deployed dual hedge does);
5. (t, gap, α, twin) chosen on the other folds (koz3.nested); per-fold change; per-site robust acceptance (k*).

  python scripts/li_dome_fusion_oof.py [OOF_SPEC]
Accept only if nested honest > deployed, no outer fold regresses, k* ≥ 3 and bootstrap P(gain>0) ≥ 0.8.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "experiments", "claude"))
DEFAULT = "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst"
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(spec=DEFAULT):
    import lightgbm as lgb
    from shapely.geometry import Point, mapping
    from shapely.strtree import STRtree
    from pyproj import Transformer
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from robust_accept import site_counts, report
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = lc.load(spec)
    sites = sorted(r for r in P if os.path.exists(os.path.join(ROOT, "data", "las_prep", r + ".npz")))
    fold = {r: items[r]["fold"] for r in sites}; folds = sorted(set(fold.values()))
    import li_dome_features as LF
    from multiprocessing import Pool
    with Pool(3) as pool:
        parts = pool.map(LF.site, sites)
    C = pd.concat([q for q in parts if len(q)], ignore_index=True)
    sites = sorted(set(C.region) & set(sites))
    import json as _json
    C["crs"] = C.region.map(lambda r: "EPSG:" + _json.load(open(os.path.join(LF.D, r, "UTM.json")))["crs"].split(":")[-1])
    FEATS = sum(LF.FAMILIES.values(), [])
    best = C[C.hit >= 0].sort_values("depth", ascending=False).groupby(["region", "hit"]).head(1).index
    C["y_"] = 0; C.loc[best, "y_"] = 1
    train_ok = (C.hit < 0) | C.index.isin(best)          # other candidates inside a mound: not trained on
    C["fold"] = C.region.map(fold); C["p"] = 0.0
    for f in folds:
        tr = (C.fold != f) & train_ok
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(C.loc[tr, FEATS], C.loc[tr, "y_"])
        te = C.fold == f
        C.loc[te, "p"] = m.predict_proba(C.loc[te, FEATS])[:, 1]
    print(f"{len(sites)} sites, {len(C)} candidates, {int(C.y_.sum())} mound positives", flush=True)
    to3857 = {}
    for r in sites:
        m = C.region == r; crs = C.loc[m, "crs"].iloc[0]
        X, Y = Transformer.from_crs(crs, 3857, always_xy=True).transform(C.loc[m, "x"].values, C.loc[m, "y"].values)
        C.loc[m, "X"] = X; C.loc[m, "Y"] = Y
    base = {}
    for r in sites:
        it = items[r]
        base[r] = decode_site_v2(P[r].astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05,
                                 comp_thr=0.5, prior={"R": 200.0, "K": 3, "snb": 0.2})

    def make_site(st, r):
        feats = list(base[r])
        if st is None:
            return feats
        t, gap, alpha, twin = st
        kur = [f for f in feats if f["properties"]["class_name"] in K]
        from shapely.geometry import shape
        cents = np.array([[shape(f["geometry"]).centroid.x, shape(f["geometry"]).centroid.y] for f in kur]) if kur else np.zeros((0, 2))
        Q = C[(C.region == r) & (C.p >= t)]
        for q in Q.itertuples():
            k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(q.Y / 6378137.0)) - np.pi / 2)))
            if len(cents) and np.hypot(cents[:, 0] - q.X, cents[:, 1] - q.Y).min() < gap * k:
                continue
            g = mapping(Point(q.X, q.Y).buffer(1.0 * k, 12))
            feats.append({"type": "Feature", "geometry": g, "properties": {"class_name": K[0], "confidence": float(alpha * q.p), "region_name": r}})
            if twin:
                feats.append({"type": "Feature", "geometry": g, "properties": {"class_name": K[1], "confidence": float(0.99 * alpha * q.p), "region_name": r}})
        return feats

    S = [None] + [(t, gap, a, tw) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for gap in (4.0, 8.0) for a in (0.5, 1.0) for tw in (False, True)]
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    fc = lambda st, f: {"type": "FeatureCollection", "features": sum((make_site(st, r) for r in sites if fold[r] == f), [])}
    Cn = fold_counts(fc, S, folds, gt, CLS)
    h, ch, o, os_, tot = nested(Cn)
    dep = weighted(_add_all(Cn[None].values()))
    print(f"deployed {dep:.4f} | nested honest {h:.4f} ({h - dep:+.4f}) | oracle {o:.4f} @ {os_} | chosen {ch}")
    for f in folds:
        print(f"   fold {f}: deployed {weighted(Cn[None][f]):.4f} → chosen {weighted(Cn[ch[f]][f]):.4f}")
    for c, v in tot.items():
        if v[0] + v[2]:
            print(f"   {c:24s} TP {v[0]} FP {v[1]} FN {v[2]}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    st = max(set(ch.values()), key=list(ch.values()).count)
    CS = site_counts(lambda s, r: {"type": "FeatureCollection", "features": make_site(s, r)}, (None, st), sites, gtr, CLS)
    report(f"dome rescue {st}", CS[None], CS[st])


if __name__ == "__main__":
    main(*sys.argv[1:2])
