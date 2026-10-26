"""Per-region score recalibration on satellite (Vesuvius per-fragment normalisation), nested + robust, deployed v14
pipeline: fused kurgan scores of a region × f = clip(c / m, lo, cap), m = mean of the region's top-k fused kurgan
scores (k 5), before the class thresholds / framing prior / hedge.
  python scripts/sp_site_calib_oof.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_fusion4_oof import gate
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    from robust_accept import site_counts, report
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
    M = models(extra=list(names[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()])); RF = region_folds()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in names]); F0 = F0[F0.score >= 0.05].reset_index(drop=True)
    far = []
    for r in F0.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        far.append((not d[1]) and d[0] > 0.5)
    F0["far"] = far
    kk = F0[F0.cls.isin(K)].sort_values("score", ascending=False).groupby("region").score.apply(lambda s: s.head(5).mean())

    def build(st, regs):
        G = F0[F0.region.isin(regs)].copy()
        if st[0] != "base":
            c, lo, cap = st
            f = G.region.map(lambda r: float(np.clip(c / max(kk.get(r, c), 1e-3), lo, cap)))
            isk = G.cls.isin(K)
            G.loc[isk, "score"] = np.minimum(G.score[isk] * f[isk], 1.0)
            G.loc[isk, "s_kp"] = np.minimum(G.s_kp[isk] * f[isk], 1.0); G.loc[isk, "s_kt"] = np.minimum(G.s_kt[isk] * f[isk], 1.0)
        G = G[(G.score >= np.where(G.cls == "arkhitektury", 0.15, 0.25)) & (~G.far | (G.score >= 0.35))]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in G[G.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
        return pd.concat([G[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    folds = sorted({RF[r] for r in regions})
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    SS = [("base",)] + list(itertools.product((0.4, 0.55, 0.7), (1.0, 0.75), (1.5, 2.5)))
    C = fold_counts(lambda st, f: fc(gate(build(st, [r for r in regions if RF[r] == f]), S[S.region.map(RF) == f])), SS, folds, gt)
    for st in SS:
        tot = _add_all(C[st].values())
        print(f"{st}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  kt {tuple(int(v) for v in tot[K[1]])}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    CS = site_counts(lambda s, r: fc(gate(build(s, [r]), S[S.region == r])), (SS[0], os_), regions, gtr)
    report(f"satellite region recalibration {os_}", CS[SS[0]], CS[os_])


if __name__ == "__main__":
    main()
