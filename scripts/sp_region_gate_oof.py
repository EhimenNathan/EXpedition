"""Region-level kurgan gate (satellite), nested + robust. Data: 269 of 671 far FPs are in 52 regions with no kurgan
labels; those regions emit fewer kurgans (median 3 vs 9) at lower peak confidence (0.45 vs 0.63). Gate: drop all
kurgan points of a region when (number of kurgan points < n_min AND their max score < s_max) — deployed v14 pipeline
otherwise (fusion, thresholds, framing prior, hedge, hillfort gate).
  python scripts/sp_region_gate_oof.py
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
    F = fuse_cls([M[n][M[n].region.isin(regions)] for n in names])
    F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True); keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
    F = F[np.array(keep)].reset_index(drop=True)
    kk = F[F.cls.isin(K)].groupby("region").score.agg(["size", "max"])

    def build(st, regs):
        G = F[F.region.isin(regs)]
        if st[0] == "cls":                # kurgan gate 0.45 + gate g for class c (drop all c points if region max < g)
            c_, g_ = st[1], st[2]; weak = set(kk[kk["max"] < 0.45].index)
            G = G[~(G.cls.isin(K) & G.region.isin(weak))]
            cm = G[G.cls == c_].groupby("region").score.max()
            G = G[~((G.cls == c_) & G.region.isin(set(cm[cm < g_].index)))]
        elif st[0] == "rel":                # gate 0.45 + relative threshold: kurgans need score ≥ rho · region max
            rho = st[1]; weak = set(kk[kk["max"] < 0.45].index)
            G = G[~(G.cls.isin(K) & G.region.isin(weak))]
            mx = G.region.map(kk["max"]).fillna(1.0)
            G = G[~(G.cls.isin(K) & (G.score < rho * mx))]
        elif st[0] != "base":
            n_min, s_max = st
            weak = set(kk[(kk["size"] < n_min) & (kk["max"] < s_max)].index)
            G = G[~(G.cls.isin(K) & G.region.isin(weak))]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in G[G.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
        return pd.concat([G[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    folds = sorted({RF[r] for r in regions})
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    SS = [("base",), (99, 0.45)] + [("cls", c, g) for c in ("arkhitektury", "gorodishcha", "fortifikatsii") for g in (0.2, 0.3, 0.4, 0.5)]
    C = fold_counts(lambda st, f: fc(gate(build(st, [r for r in regions if RF[r] == f]), S[S.region.map(RF) == f])), SS, folds, gt)
    for st in SS:
        tot = _add_all(C[st].values())
        print(f"{st}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  arch {tuple(int(v) for v in tot.get('arkhitektury', (0, 0, 0)))}  gor {tuple(int(v) for v in tot.get('gorodishcha', (0, 0, 0)))}  fort {tuple(int(v) for v in tot.get('fortifikatsii', (0, 0, 0)))}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    ref = (99, 0.45)
    if os_ != ref:
        CS = site_counts(lambda s, r: fc(gate(build(s, [r]), S[S.region == r])), (ref, os_), regions, gtr)
        report(f"class gate {os_} vs kurgan gate only", CS[ref], CS[os_])


if __name__ == "__main__":
    main()
