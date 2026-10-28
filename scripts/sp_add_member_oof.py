"""Add a new satellite fusion member (work/spor/oof_det_<name>_cands.parquet, e.g. from
experiments/claude/sp_cnn/export_member.py) to the EXACT deployed v21 / v22 satellite decoding, nested + robust.

Deployed: koz3.sp_fuse.fuse of the six members with equal weights (identical to sp_dual_oof.fuse_cls), fusion 0.25 /
architecture 0.15, framing prior (x 0.5, t2 0.35), damaged / intact hedge 0.4, hillfort scene gate 0.1 + snap (ρ 0.2,
s 0.02). Settings: weight of the new member w ∈ {0 → the six members only = deployed, 0.5, 1.0}; everything else
unchanged. Chosen on the other folds; per-fold change; per-region robust acceptance.
  python scripts/sp_add_member_oof.py <name> [--self-test]
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
WEIGHTS = (0.0, 0.5, 1.0)


def fused(M, new, w):
    """Deployed fusion of the six members, plus the new member at weight w (w = 0 → new member not used at all)."""
    from koz3.sp_fuse import fuse
    if w == 0:
        return fuse([M[n] for n in NAMES], np.ones(len(NAMES)))
    return fuse([M[n] for n in NAMES] + [new], np.r_[np.ones(len(NAMES)), w])


def self_test():
    """fuse with equal weights reproduces sp_dual_oof.fuse_cls (the deployed fusion) exactly; a new member at w > 0 only
    changes scores where it detects something."""
    from sp_dual_oof import fuse_cls
    rng = np.random.default_rng(0)
    mk = lambda n: pd.DataFrame({"region": rng.choice(["A", "B"], n), "cls": rng.choice(list(K) + ["gorodishcha"], n),
                                 "score": rng.uniform(0.05, 0.9, n), "x": rng.uniform(0, 300, n), "y": rng.uniform(0, 300, n) + 7.5e6})
    M = {n: mk(30) for n in NAMES}
    key = lambda d: d.sort_values(["region", "cls", "x", "y"]).reset_index(drop=True)
    a, b = key(fused(M, None, 0.0)), key(fuse_cls([M[n] for n in NAMES]))
    assert len(a) == len(b) and np.allclose(a.score, b.score) and np.allclose(a[["s_kp", "s_kt"]], b[["s_kp", "s_kt"]])
    far = pd.DataFrame({"region": ["A"], "cls": [K[0]], "score": [0.9], "x": [5000.0], "y": [7.6e6]})   # isolated new object
    c = fused(M, far, 1.0)
    iso = c[(c.x == 5000.0)]
    assert len(c) == len(a) + 1 and len(iso) == 1 and abs(iso.score.iloc[0] - 0.9 / 7) < 1e-9, iso
    print("self-test PASS: equal-weight fuse == deployed fuse_cls; new member adds its own objects")


def main(name):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from robust_accept import site_counts, report
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    new = pd.read_parquet(os.path.join(ROOT, f"work/spor/oof_det_{name}_cands.parquet"))[["region", "cls", "score", "x", "y"]]
    missing = sorted(set(regions) - set(new.region))
    print(f"{len(regions)} regions; new member '{name}': {len(new)} candidates, regions without any: {len(missing)}", flush=True)
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; new = new[new.region.isin(regions)]
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0
    F_by_w = {w: fused(M, new, w).reset_index(drop=True) for w in WEIGHTS}

    def pipeline(w, regs):
        """Deployed satellite decoding on the fusion with new-member weight w."""
        F0 = F_by_w[w]; F = F0[F0.region.isin(regs)]
        GC = F[(F.cls == "gorodishcha") & (F.score >= 0.02)]
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

    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    C = fold_counts(lambda w, f: fc(pipeline(w, [r for r in regions if RF[r] == f])), WEIGHTS, folds, gt)
    for w in WEIGHTS:
        tot = _add_all(C[w].values())
        print(f"w={w}: pooled {weighted(tot):.4f}  folds [{' '.join(f'{weighted(C[w][f]):.4f}' for f in folds)}]  "
              + " ".join(f"{c[:8]} {tuple(int(v) for v in tot[c])}" for c in tot), flush=True)
    h, ch, o, os_, _ = nested(C); base = weighted(_add_all(C[0.0].values()))
    print(f"deployed (6 members) {base:.4f} | nested honest {h:.4f} ({h - base:+.4f}) | best fixed {o:.4f} @ w={os_} | chosen {ch}")
    print(f"worst outer-fold change {min(weighted(C[ch[f]][f]) - weighted(C[0.0][f]) for f in folds):+.4f}")
    w = max(set(ch.values()), key=list(ch.values()).count)
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    CS = site_counts(lambda s, r: fc(pipeline(s, [r])), (0.0, w if w else 1.0), regions, gtr)
    report(f"+ member {name} w={w if w else 1.0}", CS[0.0], CS[w if w else 1.0])


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        self_test(); main(sys.argv[1])
