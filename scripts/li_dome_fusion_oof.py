"""LiDAR physics-dome rescue fused into the deployed CNN blend, nested + robust (decisive test for
experiments/claude/li_dome_detector.py).

Evidence so far (13 sites, leave-site-out, CPU): physical dome candidates (σ=5 m local-relief maxima, radial symmetry,
half-height radius) ranked by a LightGBM on physical features reach AUC 0.877 and pooled kurgan F1 0.430 alone; on
CNN-weak sites alone they match or beat the CNN (021 Novenkoe F1 0.447, 036 Timerevo 0.486, 011 Erentuna 56/69
found). Here they rescue only where the deployed CNN blend has no kurgan within `gap` metres.

Steps (no GPU):
1. candidates for every LiDAR site from data/las_prep/<region>.npz (DTM → 0.5 m), cached in work/li_dome/;
2. physical ranker trained per LiDAR fold on the other folds (labels: candidate inside a kurgan label);
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
FEATS = ["h", "snr", "sym", "rad", "l2", "rough", "dens"]
R = 0.5


def site_candidates(region, cache_dir):
    """Dome candidates (UTM x, y + physical features) from the prepared DTM; cached."""
    from scipy import ndimage as ndi
    from koz3 import dtm as dtm_mod
    from li_dome_detector import candidates
    f = os.path.join(cache_dir, region + ".parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    d, v, (x0, ytop, res), crs = dtm_mod.load(os.path.join(ROOT, "data", "las_prep", region + ".npz"))
    k = int(round(R / res))
    if k > 1:                                                   # 0.25 m → 0.5 m block mean
        H, W = (d.shape[0] // k) * k, (d.shape[1] // k) * k
        d = d[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3)); v = v[:H, :W].reshape(H // k, k, W // k, k).all((1, 3))
    C, rough, area = candidates(d.astype(np.float32), v)
    df = pd.DataFrame(C, columns=["px", "py", "h", "snr", "sym", "rad", "l2"])
    df["x"] = x0 + (df.px + 0.5) * R; df["y"] = ytop - (df.py + 0.5) * R
    df["rough"] = rough; df["dens"] = (df.snr >= 5).sum() / max(area, 1e-6); df["region"] = region; df["crs"] = crs
    os.makedirs(cache_dir, exist_ok=True); df.to_parquet(f)
    return df


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
    cache = os.path.join(ROOT, "work", "li_dome")
    C = pd.concat([site_candidates(r, cache) for r in sites], ignore_index=True)
    # labels per candidate (inside a kurgan label of its own site), in the site's UTM CRS
    C["y_"] = 0
    for r in sites:
        lab = L[(L.region == r) & L.class_name.isin(K)]
        if not len(lab):
            continue
        m = C.region == r; crs = C.loc[m, "crs"].iloc[0]
        polys = list(lab.to_crs(crs).geometry); tree = STRtree(polys)
        C.loc[m, "y_"] = [int(any(polys[j].contains(Point(x, y)) for j in tree.query(Point(x, y))))
                          for x, y in zip(C.loc[m, "x"], C.loc[m, "y"])]
    C["fold"] = C.region.map(fold); C["p"] = 0.0
    for f in folds:
        tr = C.fold != f
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, verbose=-1).fit(C.loc[tr, FEATS], C.loc[tr, "y_"])
        C.loc[~tr, "p"] = m.predict_proba(C.loc[~tr, FEATS])[:, 1]
    print(f"{len(sites)} sites, {len(C)} candidates, {int(C.y_.sum())} inside kurgan labels", flush=True)
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
