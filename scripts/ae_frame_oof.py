"""Ae framing prior (nested, Ae site folds): sheets are framed on the site, so kurgan detections farther than x·half-size
from the sheet centre need score ≥ t2 (t2 1 = dropped); base threshold t. Optional cluster condition (≥ k neighbours
with score ≥ 0.5 within 100 m) for far detections. Sources: SpOr D-FINE on Ae (v9 path) / Ae FCOS / joint D-FINE."""
import os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_fusion_oof import fc
from ae_det_oof import ae_folds, f1
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(srcs):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    RF = ae_folds(); folds = sorted(set(RF.values()))
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "Ae") & L.region.isin(RF)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    E = pd.read_parquet(os.path.join(ROOT, "work/ae/sheet_extent.parquet"))
    for name in srcs:
        D0 = pd.read_parquet(os.path.join(ROOT, f"work/ae/oof_{name}_cands.parquet")); D0 = D0[D0.cls.isin(K)].reset_index(drop=True)
        D0 = D0.join(E[["cx", "cy", "half"]], on="region"); D0["d_rel"] = np.hypot(D0.x - D0.cx, D0.y - D0.cy) / D0.half
        nb = np.zeros(len(D0))
        for reg, g in D0.groupby("region"):
            st = g[g.score >= 0.5]
            if len(st):
                t = cKDTree(st[["x", "y"]].values); nb[D0.index.get_indexer(g.index)] = [len(t.query_ball_point(p, 100.0)) for p in g[["x", "y"]].values]
        D0["nb"] = nb
        S = [(t, 9, 0, 0) for t in (0.3, 0.5, 0.7)] + [(t, x, t2, k) for t in (0.3, 0.5) for x in (0.25, 0.4, 0.6) for t2 in (0.7, 0.85, 1.0) for k in (0, 3)]
        res = {}
        for s in S:
            t, x, t2, k = s
            D = D0[D0.score >= t]
            if x < 9:
                far = D.d_rel > x
                ok = (D.score >= t2) | ((D.nb >= k + 1) & (k > 0) & (D.score >= 0.5))
                D = D[~far | ok]
            for f in folds:
                _, rr = score(fc(D[D.region.map(RF) == f][["region", "cls", "score", "x", "y"]]), gt[f])
                res[(s, f)] = np.sum([(rr[c]["tp"], rr[c]["fp"], rr[c]["fn"]) for c in K if c in rr], 0)
        tot = lambda s: np.sum([res[(s, f)] for f in folds], 0)
        P = lambda v: v[0] / max(v[0] + v[1], 1)
        for s in S[:3]:
            v = tot(s); print(f"{name:7s} no prior t={s[0]}: tp/fp/fn {tuple(int(q) for q in v)} P {P(v):.2f} F1 {f1(*v):.3f}")
        for s in sorted(S[3:], key=lambda s: -f1(*tot(s)))[:3]:
            v = tot(s); print(f"{name:7s} {s}: tp/fp/fn {tuple(int(q) for q in v)} P {P(v):.2f} F1 {f1(*v):.3f} (pooled best)")
        nv = np.zeros(3); ch = []
        for f in folds:
            s = max(S, key=lambda s_: f1(*np.sum([res[(s_, g)] for g in folds if g != f], 0))); ch.append(s); nv += res[(s, f)]
        print(f"{name:7s} NESTED tp/fp/fn {tuple(int(q) for q in nv)} P {P(nv):.2f} F1 {f1(*nv):.3f} chosen {ch}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["spdet", "ae_fcos_fast"])
