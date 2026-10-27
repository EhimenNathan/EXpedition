"""Organiser rule "mounds inside settlements are not labelled" applied to the deployed LiDAR blend, nested + robust.

Evidence (training labels): in 87 regions with hillfort labels only 7 of 974 kurgan labels lie inside a hillfort (all
at 021 Novenkoe), while 44 of 146 fortifications do (WWII objects are labelled there). Physics mound candidates with
ranker p ≥ 0.5 inside LiDAR hillforts are real kurgans 1 time in 48 (2 %), outside 62 in 139 (45 %). A kurgan
prediction inside a hillfort is therefore almost always a false alarm (break-even ≈ 14 %).

Rule tested: decode the deployed blend exactly as deployed, build the predicted hillfort extent from the model's own
hillfort mask channel (prob ≥ m, connected components ≥ gor_min_px, dilated by d metres), and drop kurgan predictions
whose centroid falls inside it. Fortifications and hillforts are untouched.
  setting None              deployed
  ("oracle",)               mask = true hillfort labels (ceiling, not deployable)
  (m, d)                    m ∈ {0.3, 0.5, 0.7}, d ∈ {0, 10, 25} m
(m, d) chosen on the other folds (koz3.nested); per-fold change; per-site robust acceptance.

  python scripts/li_hillfort_kurgan_mask_oof.py [OOF_SPEC]
Accept only if nested honest > deployed, no outer fold regresses, k* ≥ 3, bootstrap P(gain>0) ≥ 0.8.
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
DEFAULT = "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst"
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(spec=DEFAULT):
    from scipy import ndimage as ndi
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from pyproj import Transformer
    from koz3.li_train import labels, site_items, decode_site_v2, CLS, N_HM
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from robust_accept import site_counts, report
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = lc.load(spec)
    sites = sorted(P); fold = {r: items[r]["fold"] for r in sites}; folds = sorted(set(fold.values()))
    gor_min_px = 600
    base, cent, gmask = {}, {}, {}
    for r in sites:
        it = items[r]; meta = dict(it["meta"], region=r); p = P[r].astype(np.float32)
        base[r] = decode_site_v2(p, it["valid"], meta, thr=0.15, dual=0.05, comp_thr=0.5, prior={"R": 200.0, "K": 3, "snb": 0.2})
        tr = Transformer.from_crs(3857, meta["crs"], always_xy=True).transform
        rc = []
        for i, f in enumerate(base[r]):
            if f["properties"]["class_name"] in K:
                c = shape(f["geometry"]).centroid; x, y = tr(c.x, c.y)
                rc.append((i, int((meta["ytop"] - y) / meta["res"]), int((x - meta["x0"]) / meta["res"])))
        cent[r] = rc
        gmask[r] = p[N_HM + CLS.index("gorodishcha")]
    H, W = {}, {}

    def predicted_mask(r, m, d):
        key = (r, m, d)
        if key not in H:
            pm = gmask[r] >= m
            lab, n = ndi.label(pm)
            if n:
                sizes = ndi.sum(pm, lab, np.arange(1, n + 1)); keep = np.flatnonzero(sizes >= gor_min_px) + 1
                pm = np.isin(lab, keep)
            if d > 0 and pm.any():
                pm = ndi.binary_dilation(pm, iterations=max(1, int(round(d / items[r]["meta"]["res"]))))
            H[key] = pm
        return H[key]

    def oracle_mask(r):
        if r not in W:
            g = L[(L.region == r) & (L.class_name == "gorodishcha")]
            W[r] = unary_union(list(g.to_crs(3857).geometry)) if len(g) else None
        return W[r]

    def make_site(st, r):
        feats = base[r]
        if st is None:
            return feats
        drop = set()
        if st == ("oracle",):
            U = oracle_mask(r)
            if U is not None:
                drop = {i for i, f in enumerate(feats) if f["properties"]["class_name"] in K and U.contains(shape(f["geometry"]).centroid)}
        else:
            pm = predicted_mask(r, *st)
            h, w = pm.shape
            drop = {i for i, rr, cc in cent[r] if 0 <= rr < h and 0 <= cc < w and pm[rr, cc]}
        return [f for i, f in enumerate(feats) if i not in drop]

    S = [None] + [(m, d) for m in (0.3, 0.5, 0.7) for d in (0.0, 10.0, 25.0)]
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    fc = lambda st, f: {"type": "FeatureCollection", "features": sum((make_site(st, r) for r in sites if fold[r] == f), [])}
    C = fold_counts(fc, S + [("oracle",)], folds, gt, CLS)
    for st in S + [("oracle",)]:
        tot = _add_all(C[st].values())
        print(f"{str(st):14s} pooled {weighted(tot):.4f}  " + " ".join(f"{c[:8]} {tuple(int(v) for v in tot[c])}" for c in K if c in tot), flush=True)
    Cn = {s: C[s] for s in S}
    h, ch, o, os_, _ = nested(Cn); dep = weighted(_add_all(C[None].values()))
    print(f"deployed {dep:.4f} | nested honest {h:.4f} ({h - dep:+.4f}) | oracle-setting {o:.4f} @ {os_} | "
          f"label-mask ceiling {weighted(_add_all(C[('oracle',)].values())):.4f} | chosen {ch}")
    for f in folds:
        print(f"   fold {f}: deployed {weighted(C[None][f]):.4f} → chosen {weighted(C[ch[f]][f]):.4f}")
    st = max(set(ch.values()), key=list(ch.values()).count)
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    CS = site_counts(lambda s, r: {"type": "FeatureCollection", "features": make_site(s, r)}, (None, st), sites, gtr, CLS)
    report(f"hillfort kurgan mask {st}", CS[None], CS[st])


if __name__ == "__main__":
    main(*sys.argv[1:2])
