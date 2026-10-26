"""Region-majority kurgan class (satellite), nested + robust: 66 % of satellite kurgans sit in regions that are ≥ 90 %
one class. On the deployed v14 pipeline, per region the score-weighted damaged share q of the fused kurgan detections
is computed; if q ≥ θ every intact detection is relabelled damaged (mode "relabel") or additionally emitted as damaged
(mode "hedge"), symmetric for q ≤ 1 − θ. Requires ≥ n kurgan detections in the region.
  python scripts/sp_regclass_oof.py
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
    from sp_fusion4_oof import gate
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    from robust_accept import site_counts, report
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
    M = models(extra=list(names[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()])); RF = region_folds()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F = fuse_cls([M[n][M[n].region.isin(regions)] for n in names])
    F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True); keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
    F = F[np.array(keep)].reset_index(drop=True)
    k = F[F.cls.isin(K)]
    q = k.assign(w=k.score * (k.cls == K[0])).groupby("region").w.sum() / k.groupby("region").score.sum()
    nk = k.groupby("region").size()

    def build(st, regs):
        mode, th, n = st
        G = F[F.region.isin(regs)].copy(); ex = []
        for r in G[G.cls.isin(K)].itertuples():
            o = K[1] if r.cls == K[0] else K[0]
            if (r.s_kt if o == K[1] else r.s_kp) >= 0.4:
                ex.append((r.region, o, r.score * 0.99, r.x, r.y))
        G = pd.concat([G[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        if mode != "base":
            add = []
            for reg in set(G.region):
                if nk.get(reg, 0) < n:
                    continue
                maj = K[0] if q[reg] >= th else (K[1] if q[reg] <= 1 - th else None)
                if maj is None:
                    continue
                mino = K[1] if maj == K[0] else K[0]
                idx = G.index[(G.region == reg) & (G.cls == mino)]
                if mode == "relabel":
                    G.loc[idx, "cls"] = maj
                else:
                    add.append(G.loc[idx].assign(cls=maj, score=G.loc[idx].score * 0.99))
            if add:
                G = pd.concat([G] + add, ignore_index=True)
            # identical (cls, position) duplicates from relabel + hedge collapse to one
            G = G.sort_values("score", ascending=False).drop_duplicates(["region", "cls", "x", "y"])
        return G
    folds = sorted({RF[r] for r in regions})
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    SS = [("base", 0, 0)] + [(m, th, n) for m in ("relabel", "hedge") for th in (0.7, 0.8, 0.9) for n in (3, 8)]
    C = fold_counts(lambda st, f: fc(gate(build(st, [r for r in regions if RF[r] == f]), S[S.region.map(RF) == f])), SS, folds, gt)
    for st in SS:
        tot = _add_all(C[st].values())
        print(f"{st}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  kt {tuple(int(v) for v in tot[K[1]])}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    for st in (os_,):
        CS = site_counts(lambda s, r: fc(gate(build(s, [r]), S[S.region == r])), (SS[0], st), regions, gtr)
        report(f"region-majority {st}", CS[SS[0]], CS[st])


if __name__ == "__main__":
    main()
