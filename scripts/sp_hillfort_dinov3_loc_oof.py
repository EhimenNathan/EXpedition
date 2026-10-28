"""Satellite hillfort LOCALISATION with DINOv3-SAT patch probes in the deployed v21 satellite decoding, nested + robust.

Evidence (experiments/claude/sp_hillfort_dinov3_eval.py, CPU, 175 scenes, out of fold over 20 km site clusters): the
argmax of a DINOv3-SAT patch-probe map (512 + 768 px whole-scene views, PCA 32/64/128 × C ensemble, σ 0.5) falls inside
the hillfort in 0.691 of the 55 hillfort scenes vs 0.364 for the scene centre (nested single setting 0.673; every
variant 0.60–0.69). Out-of-fold points: experiments/claude/outputs/hf_dinov3_loc_points_oof.csv (one row per scene;
every row predicted by a probe that never saw that region's 20 km cluster).

Settings (point, m, gate score, gate threshold g), chosen on the other folds:
  point  "snap": deployed scene centre snapped to the best fused hillfort candidate within 0.2·half (s ≥ 0.02);
         "dino": the DINOv3 point when its map max ≥ m, else the snap point
  gate   "v21": the deployed scene classifier p; "mean3": mean(p, DINOv3 map max, DINOv3 scene classifier p_scene);
         "mean2": mean(map max, p_scene) — the gate decides which scenes emit a hillfort (confidence = gate score) and,
         as deployed, drops detector hillforts in regions with a scene below g
  ("snap", 0, "v21", 0.1) reproduces deployment.
chosen on the other folds (koz3.nested); per-fold change; per-region robust acceptance.
  python scripts/sp_hillfort_dinov3_loc_oof.py [--self-test]
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
POINTS = os.path.join(ROOT, os.environ.get("LOC_POINTS", "experiments/claude/outputs/hf_dinov3_loc_points_oof.csv"))
# (point, min map max for the DINOv3 point, gate score, gate threshold); ("snap", 0, "v21", 0.1) = deployed
SETTINGS = [("snap", 0.0, "v21", 0.1)] + [(pt, m, gs, g) for pt, m in (("snap", 0.0), ("dino", 0.0), ("dino", 0.7))
                                          for gs in ("v21", "mean3", "mean2") for g in (0.1, 0.2, 0.3, 0.4)
                                          if (pt, m, gs, g) != ("snap", 0.0, "v21", 0.1)]


BASE = ("snap", 0.0, "v21", 0.1)


def gate_score(p, loc_row, src):
    """Scene hillfort score: v21 scene classifier p, or its mean with the DINOv3 localiser peak and scene classifier."""
    if src == "v21" or loc_row is None or not np.isfinite(loc_row[3]):
        return p
    _, _, mx, ps = loc_row
    return (p + mx + ps) / 3 if src == "mean3" else (mx + ps) / 2


def match_points(Sg, Dp):
    """Sg row index → (x, y, map_max, p_scene) of the localiser point of the same region whose scene centre is nearest
    (no DataFrame merge: Sg and the point table both carry x / y3857-like columns)."""
    out = {}
    ps = Dp.p_scene.values if "p_scene" in Dp else np.full(len(Dp), np.nan)
    for r in Sg.itertuples():
        m = (Dp.region == r.region).values
        if m.any():
            d = Dp[m]; j = np.argmin(np.hypot(d.cx3857.values - r.x, d.cy3857.values - r.y3857))
            out[r.Index] = (float(d.x3857.values[j]), float(d.y3857.values[j]), float(d.map_max.values[j]), float(ps[m][j]))
    return out


def hillfort_rows(Sx, GC, half, loc, st):
    """Hillfort points for the scenes of Sx under setting st = (point, m, gate score, gate threshold); the scene
    confidence is its gate score."""
    rows = []
    pt, m_min, gsrc, g = st
    for r in Sx.itertuples():
        sc = gate_score(r.p, loc.get(r.Index), gsrc)
        if sc < g:
            continue
        cx, cy = r.x, r.y3857
        c = GC[GC.region == r.region]
        c = c[np.hypot(c.x - cx, c.y - cy) <= 0.2 * half[r.Index]]
        if len(c):
            b = c.loc[c.score.idxmax()]; cx, cy = b.x, b.y
        if pt == "snap" or r.Index not in loc:
            rows.append((r.region, "gorodishcha", sc, cx, cy)); continue
        lx, ly, mx, _ = loc[r.Index]
        rows.append((r.region, "gorodishcha", sc, lx, ly) if mx >= m_min else (r.region, "gorodishcha", sc, cx, cy))
    return rows


def self_test():
    Sg = pd.DataFrame({"region": ["A", "A", "B"], "x": [0.0, 5000.0, 0.0], "y3857": [0.0, 0.0, 0.0], "p": [0.6, 0.3, 0.05]})
    Dp = pd.DataFrame({"region": ["A", "A"], "cx3857": [10.0, 4990.0], "cy3857": [0.0, 0.0], "x3857": [100.0, 5100.0],
                       "y3857": [50.0, 0.0], "map_max": [0.95, 0.4], "p_scene": [0.9, 0.1]})
    loc = match_points(Sg, Dp)
    assert loc == {0: (100.0, 50.0, 0.95, 0.9), 1: (5100.0, 0.0, 0.4, 0.1)}, loc
    GC = pd.DataFrame({"region": ["A"], "x": [30.0], "y": [0.0], "score": [0.5]})
    half = {0: 500.0, 1: 500.0, 2: 500.0}
    snap = hillfort_rows(Sg, GC, half, loc, BASE)                     # deployed: p ≥ 0.1, snapped centre, confidence p
    assert [(r[2], r[3], r[4]) for r in snap] == [(0.6, 30.0, 0.0), (0.3, 5000.0, 0.0)], snap
    d = hillfort_rows(Sg, GC, half, loc, ("dino", 0.7, "v21", 0.1))   # scene 1: map max 0.4 < 0.7 → snap fallback
    assert [(r[3], r[4]) for r in d] == [(100.0, 50.0), (5000.0, 0.0)], d
    m3 = hillfort_rows(Sg, GC, half, loc, ("dino", 0.0, "mean3", 0.3))  # scores (0.6+0.95+0.9)/3, (0.3+0.4+0.1)/3 = 0.267 < 0.3
    assert len(m3) == 1 and abs(m3[0][2] - 2.45 / 3) < 1e-9 and m3[0][3:] == (100.0, 50.0), m3
    assert abs(gate_score(0.3, loc[1], "mean2") - 0.25) < 1e-9 and gate_score(0.05, None, "mean3") == 0.05
    print("self-test PASS: scene matching without merge, gate scores, thresholds, dino / snap fallback")


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
    loc = match_points(Sg, pd.read_csv(POINTS))
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0
    print(f"{len(regions)} regions; localiser points matched for {len(loc)} of {len(Sg)} scenes", flush=True)
    base_cache = {}

    def pipeline(st, regs):
        key = tuple(regs)
        if key not in base_cache:
            F = F0[F0.region.isin(regs)]
            F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)]; keep = []
            for r in F.itertuples():
                d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
                keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
            F = F[np.array(keep, bool)]
            ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
                  if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
            G = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
            base_cache[key] = (G, Sg[Sg.region.isin(regs)])
        G, Sx = base_cache[key]
        sc = np.array([gate_score(r.p, loc.get(r.Index), st[2]) for r in Sx.itertuples()])
        G = G[~((G.cls == "gorodishcha") & G.region.isin(set(Sx.region[sc < st[3]])))]       # as deployed, with this gate
        return pd.concat([G, pd.DataFrame(hillfort_rows(Sx, GC, half, loc, st), columns=["region", "cls", "score", "x", "y"])], ignore_index=True)

    folds = sorted({RF[r] for r in regions})
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    C = fold_counts(lambda st, f: fc(pipeline(st, [r for r in regions if RF[r] == f])), SETTINGS, folds, gt)
    for st in SETTINGS:
        tot = _add_all(C[st].values()); h = tot.get("gorodishcha", (0, 0, 0))
        print(f"{str(st):14s} pooled {weighted(tot):.4f}  hillfort (tp, fp, fn) {tuple(int(v) for v in h)}  folds "
              + " ".join(f"{weighted(C[st][f]):.4f}" for f in folds), flush=True)
    h, ch, o, os_, _ = nested(C); base = weighted(_add_all(C[BASE].values()))
    print(f"deployed (snap) {base:.4f} | nested honest {h:.4f} ({h - base:+.4f}) | best fixed {o:.4f} @ {os_} | chosen {ch}")
    print(f"worst outer-fold change {min(weighted(C[ch[f]][f]) - weighted(C[BASE][f]) for f in folds):+.4f}")
    st = max(set(ch.values()), key=list(ch.values()).count)
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    CS = site_counts(lambda s, r: fc(pipeline(s, [r])), (BASE, st), regions, gtr)
    report(f"DINOv3 hillfort localiser {st}", CS[BASE], CS[st])


if __name__ == "__main__":
    self_test()
    if sys.argv[1:2] != ["--self-test"]:
        main()
