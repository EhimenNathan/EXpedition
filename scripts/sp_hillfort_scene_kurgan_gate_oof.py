"""Scene-level labelling rule: small satellite screenshots made for a hillfort carry no kurgan labels, nested + robust.

Evidence (training satellite labels, per region): 49 regions have hillfort labels and only 6 of them any kurgan label.
Among the 42 regions whose screenshots are small (W·H < 1 Mpx), only 2 have kurgans (055 Kurmentu 39, 185 Izborsk 41);
among the 83 regions without hillforts, kurgans are common (1,182 labels). A screenshot taken for a settlement is
annotated for the settlement, so kurgan predictions inside it are mostly false alarms (break-even precision ≈ 14 %).

Rule tested on top of the exact deployed satellite decoding (as in sp_hillfort_kurgan_mask_oof.py): every kurgan
prediction is assigned to its scene (same region, smallest Chebyshev distance / half-side); if the deployed hillfort scene
classifier gives that scene p ≥ g (and the scene is small, for scope "small"), kurgan predictions with score < τ are
dropped (τ = inf drops all; τ = 0.5 keeps confident ones, protecting 055 / 185-type scenes).
  setting None                      deployed
  (g, scope, τ)                     g ∈ {0.1, 0.3, 0.5}, scope ∈ {small, all}, τ ∈ {0.5, inf}
  ("oracle",)                       drop all kurgans in regions with true hillfort labels and no kurgan labels (ceiling)
Settings chosen on the other folds (koz3.nested); per-fold change; per-region robust acceptance.
  python scripts/sp_hillfort_scene_kurgan_gate_oof.py [--self-test]
Accept only if nested honest > deployed, no outer fold regresses, k* ≥ 3, bootstrap P(gain>0) ≥ 0.8.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
SETTINGS = [(g, scope, tau) for g in (0.1, 0.3, 0.5) for scope in ("small", "all") for tau in (0.5, np.inf)]


def scene_table(Sg, geom):
    """Sg rows (scene classifier: region, x, y3857, p) → DataFrame region, cx, cy, half, small, p (geometry from the
    nearest scene_geometry entry of the same region; a region without geometry counts as one big scene)."""
    rows = []
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        cx, cy, h, sm = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1])) if g else (r.x, r.y3857, 1e6, False)
        rows.append((r.region, r.x, r.y3857, h, bool(sm), r.p))
    return pd.DataFrame(rows, columns=["region", "cx", "cy", "half", "small", "p"])


def scene_gate_drop(G, T, g, scope, tau):
    """Boolean mask over G rows: kurgans with score < tau whose own scene (row of T) has p ≥ g (and is small for
    scope "small")."""
    drop = np.zeros(len(G), bool)
    kurg = G.cls.isin(K).values & (G.score.values < tau)
    for reg, t in T.groupby("region"):
        m = np.flatnonzero((G.region.values == reg) & kurg)
        if not len(m):
            continue
        d = np.maximum(np.abs(G.x.values[m][:, None] - t.cx.values[None]), np.abs(G.y.values[m][:, None] - t.cy.values[None]))
        own = (d / np.maximum(t.half.values[None], 1.0)).argmin(1)
        hit = t.p.values[own] >= g
        if scope == "small":
            hit &= t.small.values[own]
        drop[m[hit]] = True
    return drop


def self_test():
    """Region A: a small hillfort scene (p 0.6) and a big hillfort scene (p 0.6) 5 km apart; region B: a small scene with
    p 0.05. Checks scene assignment, the p gate, the small-only scope, τ and that non-kurgan classes are never dropped."""
    T = pd.DataFrame([("A", 0.0, 0.0, 150.0, True, 0.6), ("A", 5000.0, 0.0, 2000.0, False, 0.6),
                      ("B", 0.0, 0.0, 150.0, True, 0.05)], columns=["region", "cx", "cy", "half", "small", "p"])
    G = pd.DataFrame([("A", K[0], 0.30, 20.0, 10.0),      # 0 small hillfort scene, low score
                      ("A", K[1], 0.80, -40.0, 0.0),      # 1 small hillfort scene, confident
                      ("A", K[0], 0.30, 4800.0, 100.0),   # 2 big hillfort scene
                      ("A", "gorodishcha", 0.6, 0.0, 0.0),  # 3 never dropped
                      ("B", K[0], 0.30, 10.0, 10.0)],      # 4 scene below gate
                     columns=["region", "cls", "score", "x", "y"])
    assert list(scene_gate_drop(G, T, 0.3, "small", np.inf)) == [True, True, False, False, False]
    assert list(scene_gate_drop(G, T, 0.3, "small", 0.5)) == [True, False, False, False, False]
    assert list(scene_gate_drop(G, T, 0.3, "all", np.inf)) == [True, True, True, False, False]
    assert list(scene_gate_drop(G, T, 0.01, "all", np.inf)) == [True, True, True, False, True]
    assert not scene_gate_drop(G, T, 0.7, "all", np.inf).any()
    Sg = pd.DataFrame({"region": ["A", "A"], "x": [3.0, 5010.0], "y3857": [0.0, 0.0], "p": [0.6, 0.2]})
    geom = {"A": [(0.0, 0.0, 150.0, True), (5000.0, 0.0, 2000.0, False)]}
    T2 = scene_table(Sg, geom)
    assert list(T2.small) == [True, False] and list(T2.half) == [150.0, 2000.0], T2
    print("self-test PASS: scene assignment, p gate, small-only scope, τ protection, non-kurgans untouched")


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls
    from robust_accept import site_counts, report
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    T = scene_table(Sg[Sg.region.isin(regions)], geom)
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    hf_regions = set(L[L.class_name == "gorodishcha"].region); k_regions = set(L[L.class_name.isin(K)].region)
    ORACLE = hf_regions - k_regions
    print(f"{len(regions)} regions, {len(F0)} fused candidates | scenes {len(T)} (small {int(T.small.sum())}) | "
          f"hillfort-label regions {len(hf_regions)}, of which with kurgan labels {len(hf_regions & k_regions)}", flush=True)
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0

    def deployed(regs):
        """Deployed satellite decoding (fusion 0.25 / arch 0.15, framing prior 0.5 / 0.35, hedge 0.4, gate 0.1, hillfort
        snap ρ 0.2 s 0.02), identical to sp_hillfort_kurgan_mask_oof.pipeline at ρ = 0."""
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
            c = GC[GC.region == r.region]
            c = c[np.hypot(c.x - cx, c.y - cy) <= 0.2 * half[r.Index]]
            if len(c):
                b = c.loc[c.score.idxmax()]; cx, cy = b.x, b.y
            rows.append((r.region, "gorodishcha", r.p, cx, cy))
        return pd.concat([G, pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)

    cache = {}

    def pipeline(st, regs):
        key = tuple(regs)
        if key not in cache:
            cache[key] = deployed(regs)
        G = cache[key]
        if st is None:
            return G
        if st == ("oracle",):
            return G[~(G.cls.isin(K) & G.region.isin(ORACLE)).values]
        return G[~scene_gate_drop(G, T, *st)]

    folds = sorted({RF[r] for r in regions})
    S = [None] + SETTINGS
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    C = fold_counts(lambda st, f: fc(pipeline(st, [r for r in regions if RF[r] == f])), S + [("oracle",)], folds, gt)
    Gall = pd.concat([cache[k] for k in cache], ignore_index=True)
    for st in S + [("oracle",)]:
        tot = _add_all(C[st].values())
        nd = 0 if st is None else int((Gall.cls.isin(K) & Gall.region.isin(ORACLE)).sum() if st == ("oracle",) else scene_gate_drop(Gall, T, *st).sum())
        print(f"{str(st):22s} pooled {weighted(tot):.4f}  dropped {nd:4d}  folds [{' '.join(f'{weighted(C[st][f]):.4f}' for f in folds)}]  "
              + " ".join(f"{c[:8]} {tuple(int(v) for v in tot[c])}" for c in K if c in tot), flush=True)
    Cn = {s: C[s] for s in S}
    h, ch, o, os_, _ = nested(Cn); base = weighted(_add_all(C[None].values()))
    print(f"deployed {base:.4f} | nested honest {h:.4f} ({h - base:+.4f}) | best fixed {o:.4f} @ {os_} | "
          f"oracle ceiling {weighted(_add_all(C[('oracle',)].values())):.4f} | chosen {ch}")
    for f in folds:
        print(f"   fold {f}: deployed {weighted(C[None][f]):.4f} → chosen {weighted(C[ch[f]][f]):.4f}")
    print(f"worst outer-fold change {min(weighted(C[ch[f]][f]) - weighted(C[None][f]) for f in folds):+.4f}")
    st = max(set(ch.values()), key=list(ch.values()).count)
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    CS = site_counts(lambda s, r: fc(pipeline(s, [r])), (None, st), regions, gtr)
    report(f"hillfort scene kurgan gate {st}", CS[None], CS[st])


if __name__ == "__main__":
    self_test()
    if sys.argv[1:2] != ["--self-test"]:
        main()
