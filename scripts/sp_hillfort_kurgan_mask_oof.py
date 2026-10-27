"""Organiser rule "mounds inside settlements are not labelled" for the deployed satellite pipeline, nested + robust.

Training labels contain no satellite kurgan inside a satellite hillfort label (87 regions with hillforts: 7 of 974
kurgans inside, all LiDAR at 021 Novenkoe). The deployed satellite decoding emits hillforts as points (scene gate +
snap and detector hillforts); here satellite kurgan predictions within ρ · R of an emitted hillfort point are
dropped, R = 54 m (radius of the median satellite hillfort label, 9,206 m²). λ = 0 / ρ = 0 reproduces deployment;
("oracle",) drops kurgans inside the true hillfort polygons (ceiling, not deployable).
  python scripts/sp_hillfort_kurgan_mask_oof.py
Accept only if nested honest > deployed, no outer fold regresses, k* ≥ 3, bootstrap P(gain>0) ≥ 0.8.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
RHOS = (0.0, 0.5, 0.75, 1.0, 1.5)
R_HILLFORT = 54.0


def near_hillfort(G, radius_m):
    """Boolean mask of kurgan rows within radius_m ground metres of a hillfort row of the same region
    (EPSG:3857 distances are scaled by 1/cos(latitude))."""
    drop = np.zeros(len(G), bool)
    kurg = G.cls.isin(K).values
    for r, h in G[G.cls == "gorodishcha"].groupby("region"):
        m = (G.region.values == r) & kurg
        if m.any():
            d = np.hypot(G.x.values[m][:, None] - h.x.values[None], G.y.values[m][:, None] - h.y.values[None]).min(1)
            kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(G.y.values[m] / 6378137.0)) - np.pi / 2)))
            drop[np.flatnonzero(m)[d < radius_m * kk]] = True
    return drop


def self_test():
    """At 55° N a 50 m ground distance is ≈ 87 m in EPSG:3857: kurgans 40 m and 60 m (ground) from a hillfort point with
    a 54 m radius must be dropped and kept respectively; other regions and non-kurgan classes are untouched."""
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3857, always_xy=True).transform
    lat = 55.0; m_per_deg = 111320 * np.cos(np.radians(lat))
    pts = [("A", "gorodishcha", 0.0), ("A", K[0], 40.0), ("A", K[1], 60.0), ("A", "arkhitektury", 10.0), ("B", K[0], 10.0)]
    rows = []
    for reg, cls, east_m in pts:
        x, y = tr(37.0 + east_m / m_per_deg, lat); rows.append((reg, cls, 0.5, x, y))
    G = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])
    assert list(near_hillfort(G, 54.0)) == [False, True, False, False, False], list(near_hillfort(G, 54.0))
    assert not near_hillfort(G, 0.0).any()
    print("self-test PASS: ground-metre radius at 55° N, same-region kurgans only")


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    from robust_accept import site_counts, report
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    print(f"{len(regions)} regions, {len(F0)} fused candidates")
    from shapely.geometry import Point
    from shapely.ops import unary_union
    HF = {r: unary_union(list(L[(L.region == r) & (L.class_name == "gorodishcha")].geometry)) for r in regions}
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0

    def pipeline(rho, regs):
        """Deployed satellite decoding (fusion 0.25 / arch 0.15, framing prior 0.5 / 0.35, hedge 0.4, gate 0.1, hillfort
        snap ρ 0.2 s 0.02), then kurgans near emitted hillfort points (or inside true hillforts for the oracle) removed."""
        F = F0[F0.region.isin(regs)].copy()
        F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)]; keep = []
        for r in F.itertuples():
            d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
            keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
        F = F[np.array(keep, bool)]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
        G = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        Sx = Sg[Sg.region.isin(regs)]
        G = G[~((G.cls == "gorodishcha") & G.region.isin(set(Sx[Sx.p < 0.1].region)))]
        rows = []
        for r in Sx[Sx.p >= 0.1].itertuples():
            cx, cy = r.x, r.y3857
            c = GC[(GC.region == r.region) & (GC.score >= 0.02)]
            c = c[np.hypot(c.x - cx, c.y - cy) <= 0.2 * half[r.Index]]
            if len(c):
                b = c.loc[c.score.idxmax()]; cx, cy = b.x, b.y
            rows.append((r.region, "gorodishcha", r.p, cx, cy))
        G = pd.concat([G, pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        kurg = G.cls.isin(K).values
        if rho == ("oracle",):
            drop = np.array([k and not HF[r].is_empty and HF[r].contains(Point(x, y)) for k, r, x, y in zip(kurg, G.region, G.x, G.y)], bool)
            return G[~drop]
        if rho == 0.0:
            return G
        drop = near_hillfort(G, rho * R_HILLFORT)
        return G[~drop]

    folds = sorted({RF[r] for r in regions})
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    C = fold_counts(lambda lam, f: fc(pipeline(lam, [r for r in regions if RF[r] == f])), list(RHOS) + [("oracle",)], folds, gt)
    print(f"label-mask ceiling: {weighted(_add_all(C[('oracle',)].values())):.4f}")
    C = {k: v for k, v in C.items() if k != ("oracle",)}
    for lam in RHOS:
        tot = _add_all(C[lam].values())
        per_fold = " ".join(f"{weighted(C[lam][f]):.4f}" for f in folds)
        print(f"ρ={lam:<4} pooled {weighted(tot):.4f}  folds [{per_fold}]  "
              + " ".join(f"{c[:8]} {tuple(int(v) for v in tot[c])}" for c in K if c in tot), flush=True)
    h, ch, o, os_, _ = nested(C)
    base = weighted(_add_all(C[0.0].values()))
    print(f"deployed (ρ=0) {base:.4f} | nested honest {h:.4f} ({h - base:+.4f}) oracle {o:.4f} @ λ={os_} chosen {ch}")
    worst = min(weighted(C[ch[f]][f]) - weighted(C[0.0][f]) for f in folds)
    print(f"worst outer-fold change {worst:+.4f}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    lam = max(set(ch.values()), key=list(ch.values()).count)
    CS = site_counts(lambda s, r: fc(pipeline(s, [r])), (0.0, lam), regions, gtr)
    report(f"hillfort kurgan mask ρ={lam}", CS[0.0], CS[lam])


if __name__ == "__main__":
    self_test()
    if sys.argv[1:2] != ["--self-test"]:
        main()
