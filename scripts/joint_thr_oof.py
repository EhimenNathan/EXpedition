"""Joint (pooled-metric) thresholds across data types, honest.

The official F1 is pooled per class over all data types, so a detection is worth emitting when its chance of being a
TP exceeds F1_class / (4 − F1_class) of the POOLED class — not of its own data type. Thresholds tuned per data type on
that type's OOF alone are therefore too strict wherever the type is better than the pool (LiDAR) and Ae's missed labels
lower the pooled F1 further.

Per region and setting we store per-class (tp, fp, fn) for: satellite (deployed v14 pipeline, all class thresholds × m),
LiDAR (deployed 4-model blend + neighbourhood prior, peak / component threshold t), Ae (SpOr-D-FINE on Ae, damaged
kurgans ≥ a, or off), Or (Or FCOS architecture ≥ o); geophysics fixed (val_sim). Validation-shaped draws (30 SpOr, 13 Li,
4 Ae, 2 MagGr, 1 EtGr, 1 TerFl, 1 Or) sum the counts → official weighted F1. Honest: regions of every type are split
in halves; the setting is chosen on draws from half A and scored on draws from half B (and vice versa), 10 splits.
  python scripts/joint_thr_oof.py
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from koz3.nested import weighted, _add_all  # noqa: E402

COMP = {"SpOr": 30, "Li": 13, "Ae": 4, "MagGr": 2, "EtGr": 1, "TerFl": 1, "Or": 1}
SAT_M = (0.6, 0.7, 0.8, 0.9, 1.0, 1.15)
LI_T = (0.06, 0.08, 0.1, 0.12, 0.15, 0.2)
AE_A = (None, 0.9, 0.8, 0.7)
OR_O = (0.45, 0.35, 0.25)
CACHE = os.path.join(ROOT, "work/joint_thr_counts.pkl")


def counts(fc, gt):
    from koz3.score import score
    _, res = score(fc, gt)
    return {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}


def build():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from val_sim import pts, geo_predictions
    from sp_assemble_oof import models
    from sp_fusion_oof import fc
    from sp_fusion4_oof import gate
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    from koz3.li_train import labels, site_items, decode_site_v2
    import li_compare_oof as lc
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    GT = lambda m, r: to_gt_geojson(L[(L["mod"] == m) & (L.region == r)])
    C = {}
    # ---- satellite
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
    M = models(extra=list(names[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in names]); F0 = F0[F0.score >= 0.25 * min(SAT_M) * 0.6]
    geom = scene_geometry(); far = []
    for r in F0.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        far.append((not d[1]) and d[0] > 0.5)
    F0["far"] = far
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet"))
    for m in SAT_M:
        F = F0[(F0.score >= np.where(F0.cls == "arkhitektury", 0.15, 0.25) * m) & (~F0.far | (F0.score >= 0.35 * m))]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
        G = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        for r in regions:
            C[("SpOr", r, m)] = counts(fc(gate(G[G.region == r], S[S.region == r])), GT("SpOr", r))
    print("satellite done", len(regions), flush=True)
    # ---- LiDAR
    Li = labels(); items = {it["region"]: it for it in site_items(Li)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    pri = {"R": 200.0, "K": 3, "snb": 0.2}
    for r in sorted(P["ens"]):
        b = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32); it = items[r]
        for t in LI_T:
            fs = decode_site_v2(b, it["valid"], dict(it["meta"], region=r), thr=t, dual=0.05, comp_thr=0.5, prior=pri)
            C[("Li", r, t)] = counts({"type": "FeatureCollection", "features": fs}, GT("Li", r))
    print("LiDAR done", len(P["ens"]), flush=True)
    # ---- Ae (SpOr D-FINE on Ae film, damaged kurgans)
    A = pd.read_parquet(os.path.join(ROOT, "work/ae/oof_spdet_cands.parquet")); A = A[A.cls == "kurgany_povrezhdennye"]
    for r in sorted(set(L[L["mod"] == "Ae"].region)):
        for a in AE_A:
            g = A[(A.region == r) & (A.score >= a)] if a is not None else A.iloc[:0]
            C[("Ae", r, a)] = counts({"type": "FeatureCollection", "features": pts(g, "ae")}, GT("Ae", r))
    # ---- Or (Or FCOS architecture, LORO)
    O3 = pd.read_parquet(os.path.join(ROOT, "work/ae/oof_or_fcos_fast_cands.parquet")); O3 = O3[O3.cls == "arkhitektury"]
    for r in sorted(set(L[L["mod"] == "Or"].region)):
        for o in OR_O:
            C[("Or", r, o)] = counts({"type": "FeatureCollection", "features": pts(O3[(O3.region == r) & (O3.score >= o)], "or")}, GT("Or", r))
    # ---- geophysics (fixed, val_sim)
    PP = geo_predictions()
    for m in ("MagGr", "EtGr", "TerFl"):
        for r in sorted(set(L[L["mod"] == m].region)):
            C[(m, r, None)] = counts({"type": "FeatureCollection", "features": PP.get(m, {}).get(r, [])}, GT(m, r))
    pickle.dump(C, open(CACHE, "wb"))
    return C


def main(n_draws=300, n_splits=10):
    C = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) and not os.environ.get("REBUILD") else build()
    regs = {m: sorted({k[1] for k in C if k[0] == m}) for m in COMP}
    print({m: len(v) for m, v in regs.items()})
    D = {"SpOr": 1.0, "Li": 0.15, "Ae": None, "Or": 0.45}
    grid = list(itertools.product(SAT_M, LI_T, AE_A, OR_O))

    def draws(pool, n, rng):
        return [{m: list(rng.choice(pool[m], COMP[m], replace=len(pool[m]) < COMP[m])) for m in COMP} for _ in range(n)]

    def evaluate(dr, st):
        sm, lt, aa, oo = st; set_ = {"SpOr": sm, "Li": lt, "Ae": aa, "Or": oo}; out = []
        for d in dr:
            tot = _add_all(C[(m, r, set_.get(m))] for m in COMP for r in d[m])
            out.append(weighted(tot))
        return np.mean(out)
    dep = (D["SpOr"], D["Li"], D["Ae"], D["Or"])
    rng = np.random.default_rng(0)
    full = draws(regs, n_draws, rng)
    print(f"deployed (pooled draws, all regions): {evaluate(full, dep):.4f}")
    for ax, vals, idx in (("sat×", SAT_M, 0), ("li_t", LI_T, 1), ("ae", AE_A, 2), ("or", OR_O, 3)):
        print(ax, " ".join(f"{v}:{evaluate(full, tuple(v if i == idx else dep[i] for i in range(4))):.4f}" for v in vals), flush=True)
    hon, base, ch = [], [], []
    for s in range(n_splits):
        rs = np.random.default_rng(100 + s); A_, B_ = {}, {}
        for m, v in regs.items():
            p = list(rs.permutation(v)); h = max(1, len(p) // 2)
            A_[m], B_[m] = (p[:h], p[h:] or p[:h]) if len(p) > 1 else (p, p)
        for tr, te in ((A_, B_), (B_, A_)):
            dtr, dte = draws(tr, 120, rs), draws(te, 120, rs)
            best = max(grid, key=lambda st: evaluate(dtr, st))
            hon.append(evaluate(dte, best)); base.append(evaluate(dte, dep)); ch.append(best)
        print(f"split {s}: chosen {ch[-2]} / {ch[-1]}  honest {np.mean(hon[-2:]):.4f} vs deployed {np.mean(base[-2:]):.4f}", flush=True)
    g = np.array(hon) - np.array(base)
    print(f"HONEST joint thresholds {np.mean(hon):.4f} vs deployed {np.mean(base):.4f}: gain {g.mean():+.4f} (wins {int((g > 0).sum())}/{len(g)})")
    print("most chosen:", pd.Series([str(c) for c in ch]).value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
