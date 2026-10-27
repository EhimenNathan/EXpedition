"""Satellite post-processing re-tuned under the public scene-type mix (importance-weighted nested OOF).

Drift (docs/DATA_INSIGHTS_2026-10-27.md): 15 % of training satellite regions contain large high-resolution scenes
(≥ 20 MB; median GSD 0.15 m, 5.1 labels/km²), against 57 % of public regions (17 of 30). Pooled OOF tuning is
dominated by small screenshots (0.74 m, 10.8 labels/km²). Here every region's (tp, fp, fn) counts are multiplied by
w = public share / training share of its scene type before pooling, so the selected thresholds, framing prior and
hedge reflect the composition the platform will score. Settings are chosen on the other folds (nested); the
unweighted result is printed alongside so the effect of re-weighting itself is visible.

  python scripts/sp_composition_oof.py [public_big_share=0.57]
Accept only if the weighted nested score improves on deployed AND the unweighted nested score does not regress.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
DEPLOYED = (0.25, 0.25, 0.5, 0.35, 0.4)   # thr small scenes, thr big scenes, framing x, framing t2, hedge


def big_regions():
    """Regions with at least one satellite scene ≥ 20 MB (the public 'large scene' type), from the raw data folder."""
    import glob
    from koz3.labels import DATA
    out = set()
    for p in glob.glob(os.path.join(DATA, "*", "04_*", "*.tif")):
        if os.path.getsize(p) >= 20e6:
            out.add(os.path.basename(os.path.dirname(os.path.dirname(p))))
    return out


def weighted_add(counts, w):
    tot = {}
    for r, cc in counts.items():
        for c, (tp, fp, fn) in cc.items():
            a = tot.get(c, (0.0, 0.0, 0.0))
            tot[c] = (a[0] + w[r] * tp, a[1] + w[r] * fp, a[2] + w[r] * fn)
    return tot


def main(public_big=0.57):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import weighted
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls
    from robust_accept import site_counts
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds(); BIG = big_regions() & set(regions)
    share = len(BIG) / len(regions)
    w = {r: (public_big / share if r in BIG else (1 - public_big) / (1 - share)) for r in regions}
    print(f"{len(regions)} OOF regions, {len(BIG)} with large scenes ({share:.2f}); weights big {public_big / share:.2f}, "
          f"small {(1 - public_big) / (1 - share):.2f}")
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0

    def pipeline(st, reg):
        t_s, t_b, fx, ft2, hedge = st
        F = F0[F0.region == reg]
        t = t_b if reg in BIG else t_s
        F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, t)]; keep = []
        for r in F.itertuples():
            d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
            keep.append(d[1] or d[0] <= fx or r.score >= ft2)
        F = F[np.array(keep, bool)]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= hedge]
        G = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        Sx = Sg[Sg.region == reg]
        if (Sx.p < 0.1).any():
            G = G[G.cls != "gorodishcha"]
        rows = []
        for r in Sx[Sx.p >= 0.1].itertuples():
            cx, cy = r.x, r.y3857
            c = GC[(GC.region == reg) & (np.hypot(GC.x - cx, GC.y - cy) <= 0.2 * half[r.Index])]
            if len(c):
                b = c.loc[c.score.idxmax()]; cx, cy = b.x, b.y
            rows.append((reg, "gorodishcha", r.p, cx, cy))
        return fc(pd.concat([G, pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])], ignore_index=True))

    grid = sorted({DEPLOYED} | {(ts, tb, fx, ft2, h) for ts in (0.25,) for tb in (0.25, 0.3, 0.35, 0.4)
                                for fx in (0.35, 0.5, 0.65) for ft2 in (0.35, 0.45) for h in (0.4,)})
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    CS = site_counts(pipeline, grid, regions, gtr)
    folds = sorted({RF[r] for r in regions})
    for name, wt in (("unweighted", {r: 1.0 for r in regions}), ("public-mix weighted", w)):
        score = lambda st, regs: weighted(weighted_add({r: CS[st][r] for r in regs}, wt))
        chosen, tot = {}, {}
        for f in folds:
            inner = [r for r in regions if RF[r] != f]; outer = [r for r in regions if RF[r] == f]
            best = max(grid, key=lambda st: score(st, inner)); chosen[f] = best
            for c, v in weighted_add({r: CS[best][r] for r in outer}, wt).items():
                a = tot.get(c, (0, 0, 0)); tot[c] = tuple(x + y for x, y in zip(a, v))
        dep = score(DEPLOYED, regions)
        print(f"{name:20s} deployed {dep:.4f} | nested honest {weighted(tot):.4f} ({weighted(tot) - dep:+.4f}) | chosen {list(chosen.values())}")
        for f in folds:
            outer = [r for r in regions if RF[r] == f]
            print(f"    fold {f}: deployed {score(DEPLOYED, outer):.4f} → chosen {score(chosen[f], outer):.4f}")
        big = [r for r in regions if r in BIG]; small = [r for r in regions if r not in BIG]
        print(f"    deployed by scene type: big {score(DEPLOYED, big):.4f}  small {score(DEPLOYED, small):.4f}")


if __name__ == "__main__":
    main(*[float(a) for a in sys.argv[1:2]])
