"""Hillfort point placement (satellite), nested + robust. Data: hillforts lie close to the scene centre (median 0.085 of
the half-size) but the centre is inside the polygon only 29 % of the time. The deployed gate puts a hillfort point at
the scene centre when the scene classifier p ≥ 0.1. Here that point is snapped to the best low-threshold fused
hillfort candidate (any member, score ≥ s_min) within ρ·half of the centre ("snap"), or the snapped point is added
next to the centre point ("both"). Everything else = deployed v14.
  python scripts/sp_gor_snap_oof.py
"""
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
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    from robust_accept import site_counts, report
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
    M = models(extra=list(names[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()])); RF = region_folds()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in names])
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    F = F0[F0.score >= np.where(F0.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True); keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
    F = F[np.array(keep)]
    ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
          if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
    F = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    half = {}
    for r in S.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0

    def build(st, regs):
        mode, rho, smin = st
        G = F[F.region.isin(regs)]; Sx = S[S.region.isin(regs)]
        G = G[~((G.cls == "gorodishcha") & G.region.isin(set(Sx[Sx.p < 0.1].region)))]
        rows = []
        for r in Sx[Sx.p >= 0.1].itertuples():
            cx, cy = r.x, r.y3857; pts = [(cx, cy, r.p)]
            if mode != "base":
                c = GC[(GC.region == r.region) & (GC.score >= smin)]
                c = c[np.hypot(c.x - cx, c.y - cy) <= rho * half[r.Index]]
                if len(c):
                    b = c.loc[c.score.idxmax()]
                    pts = [(b.x, b.y, r.p)] if mode == "snap" else [(cx, cy, r.p), (b.x, b.y, r.p * 0.99)]
            rows += [(r.region, "gorodishcha", p, x, y) for x, y, p in pts]
        return pd.concat([G, pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    folds = sorted({RF[r] for r in regions})
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    SS = [("base", 0, 0)] + [(m, rho, s) for m in ("snap", "both") for rho in (0.1, 0.2, 0.35) for s in (0.02, 0.1)]
    C = fold_counts(lambda st, f: fc(build(st, [r for r in regions if RF[r] == f])), SS, folds, gt)
    for st in SS:
        tot = _add_all(C[st].values())
        print(f"{st}: pooled {weighted(tot):.4f}  gor {tuple(int(v) for v in tot['gorodishcha'])}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    CS = site_counts(lambda s, r: fc(build(s, [r])), (SS[0], os_), regions, gtr)
    report(f"hillfort snap {os_}", CS[SS[0]], CS[os_])


if __name__ == "__main__":
    main()
