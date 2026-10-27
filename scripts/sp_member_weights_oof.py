"""Satellite fusion member weights by hill climbing (Caruana ensemble selection) and Nelder–Mead, nested + robust.

Deployed fusion (sp_dual_oof.fuse_cls): per region and class family, member detections within 8 m form one object whose
score is the EQUAL-weight mean of the six members' best scores (dfine, rtdetr, hm, dfnodn, balq, balq3). Equal weights
were never optimised. Clustering, leader class and the kurgan hedge scores do not depend on the weights, so objects are
built once and only score = w·best / Σw changes; the rest of the deployed decoding (fusion 0.25 / arch 0.15, framing
prior, hedge 0.4, hillfort gate 0.1 + snap) is re-run unchanged for every weight vector. With w = 1 it reproduces
deployment exactly (checked at start-up against fuse_cls).

Optimisers, each run separately for every outer fold on the OTHER folds' pooled official counts:
  hc   Caruana: start from the equal bag (one copy of each member), add the member copy that most improves the objective,
       with replacement, until nothing improves (≤ 12 additions). Weights = copy counts.
  nm   Nelder–Mead on log-weights (start 0 = equal, initial step 0.7, ≤ 60 evaluations). The objective is piecewise
       constant in w, so NM is expected to stall on plateaus; it is reported as a check, not trusted alone.
The held-out fold is scored with its own fold's weights (honest). Per-region robust acceptance uses the same
per-fold weights (each region scored with the weights chosen without its fold).
  python scripts/sp_member_weights_oof.py [--self-test]
Accept only if honest > deployed, no outer fold regresses, k* ≥ 3, bootstrap P(gain>0) ≥ 0.8.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
fam = lambda c: "k" if c in K else c


def objects(models, match_m=8.0):
    """fuse_cls clustering, keeping each member's best score per object → DataFrame region, cls, x, y, s_kp, s_kt, b0..bN."""
    allc = pd.concat([m.assign(mi=i) for i, m in enumerate(models)], ignore_index=True)
    allc["fam"] = allc.cls.map(fam); out = []
    for (reg, f), g in allc.groupby(["region", "fam"]):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        tree = cKDTree(g[["x", "y"]].values)
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            idx = [j for j in tree.query_ball_point(g.loc[i, ["x", "y"]].values, match_m * k) if not used[j]]
            used[idx] = True
            best = np.zeros(len(models))
            for j in idx:
                best[g.mi[j]] = max(best[g.mi[j]], g.score[j])
            sc = {c: float(g.score[[j for j in idx if g.cls[j] == c]].max()) if any(g.cls[j] == c for j in idx) else 0.0 for c in K}
            out.append((reg, g.cls[i], g.x[i], g.y[i], sc[K[0]], sc[K[1]], *best))
    return pd.DataFrame(out, columns=["region", "cls", "x", "y", "s_kp", "s_kt"] + [f"b{i}" for i in range(len(models))])


def weighted_score(O, w):
    B = O[[f"b{i}" for i in range(len(w))]].to_numpy()
    return B @ np.asarray(w, float) / float(np.sum(w))


def caruana(obj, n, max_add=12):
    """Greedy forward selection with replacement from the equal bag; obj(weights) → value to maximise."""
    bag = np.ones(n); cur = obj(bag); path = [(bag.copy(), cur)]
    for _ in range(max_add):
        best = max(((obj(bag + np.eye(n)[i]), i) for i in range(n)), key=lambda t: t[0])
        if best[0] <= cur + 1e-9:
            break
        bag = bag + np.eye(n)[best[1]]; cur = best[0]; path.append((bag.copy(), cur))
    return bag, cur, path


def nelder_mead(obj, n, maxfev=60, step=0.7):
    from scipy.optimize import minimize
    x0 = np.zeros(n); simplex = np.vstack([x0] + [x0 + step * np.eye(n)[i] for i in range(n)])
    r = minimize(lambda t: -obj(np.exp(t)), x0, method="Nelder-Mead",
                 options={"maxfev": maxfev, "initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-6})
    w = np.exp(r.x)
    return (w if -r.fun > obj(np.ones(n)) + 1e-9 else np.ones(n)), max(-r.fun, obj(np.ones(n)))


def self_test():
    """Objects: weight-independent clustering + w·best/Σw equals fuse_cls at equal weights; Caruana finds the planted
    best member; Nelder–Mead never returns a worse point than equal weights."""
    from sp_dual_oof import fuse_cls
    rng = np.random.default_rng(0); ms = []
    for i in range(3):
        n = 40
        ms.append(pd.DataFrame({"region": rng.choice(["A", "B"], n), "cls": rng.choice(list(K) + ["gorodishcha"], n),
                                "score": rng.uniform(0.05, 0.9, n), "x": rng.uniform(0, 300, n), "y": rng.uniform(0, 300, n) + 7.5e6}))
    O = objects(ms); F = fuse_cls(ms)
    key = lambda d: d.sort_values(["region", "cls", "x", "y"]).reset_index(drop=True)
    a, b = key(O.assign(score=weighted_score(O, np.ones(3)))), key(F)
    assert len(a) == len(b) and np.allclose(a.score, b.score) and np.allclose(a[["s_kp", "s_kt"]], b[["s_kp", "s_kt"]])
    target = np.array([0.2, 0.7, 0.1])
    obj = lambda w: -np.sum((np.asarray(w) / np.sum(w) - target) ** 2)
    bag, val, _ = caruana(obj, 3)
    assert np.argmax(bag) == 1 and val > obj(np.ones(3)), bag
    w, v = nelder_mead(obj, 3)
    assert v >= obj(np.ones(3)) and abs(obj(w) - v) < 1e-9
    flat = lambda w: 0.5                                               # plateau: must keep equal weights
    assert np.allclose(nelder_mead(flat, 3)[0], 1) and np.allclose(caruana(flat, 3)[0], 1)
    print("self-test PASS: fusion parity with fuse_cls, Caruana, Nelder–Mead safe on plateaus")


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import weighted, _add_all
    from koz3.score import score
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls
    from robust_accept import site_counts, report
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    ms = [M[n][M[n].region.isin(regions)] for n in NAMES]
    O = objects(ms)
    F0 = fuse_cls(ms)
    key = lambda d: d.sort_values(["region", "cls", "x", "y"]).reset_index(drop=True)
    assert np.allclose(key(O.assign(score=weighted_score(O, np.ones(len(NAMES))))).score, key(F0).score), "parity with fuse_cls failed"
    # weight-independent framing prior inputs
    rel, small = [], []
    for r in O.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        rel.append(d[0]); small.append(d[1])
    O["rel"], O["small"] = rel, small
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0
    print(f"{len(regions)} regions, {len(O)} fused objects; parity with deployed fusion OK", flush=True)

    def pipeline(w, regs):
        F = O[O.region.isin(regs)].assign(score=lambda d: weighted_score(d, w))
        GC = F[(F.cls == "gorodishcha") & (F.score >= 0.02)]
        F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)]
        F = F[F.small.values | (F.rel.values <= 0.5) | (F.score.values >= 0.35)]
        kf = F[F.cls.isin(K)]
        hedge = np.where(kf.cls == K[0], kf.s_kt, kf.s_kp) >= 0.4
        ex = pd.DataFrame({"region": kf.region[hedge], "cls": np.where(kf.cls[hedge] == K[0], K[1], K[0]),
                           "score": kf.score[hedge] * 0.99, "x": kf.x[hedge], "y": kf.y[hedge]})
        G = pd.concat([F[["region", "cls", "score", "x", "y"]], ex], ignore_index=True)
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

    fold_regs = {f: [r for r in regions if RF[r] == f] for f in folds}
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    memo = {}; t0 = time.time()

    def counts(w):
        """{fold: {cls: (tp, fp, fn)}} for a weight vector (scale-free, memoised)."""
        k = tuple(np.round(np.asarray(w, float) / np.sum(w), 4))
        if k not in memo:
            memo[k] = {}
            for f in folds:
                _, res = score(fc(pipeline(w, fold_regs[f])), gt[f])
                memo[k][f] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
            if len(memo) % 10 == 0:
                print(f"   {len(memo)} weight vectors scored, {time.time() - t0:.0f}s", flush=True)
        return memo[k]

    eq = np.ones(len(NAMES)); base = counts(eq)
    print(f"deployed (equal weights) {weighted(_add_all(base.values())):.4f}  folds "
          + " ".join(f"{weighted(base[f]):.4f}" for f in folds), flush=True)
    chosen = {"hc": {}, "nm": {}}
    for f in folds:
        obj = lambda w, f=f: weighted(_add_all(v for g, v in counts(w).items() if g != f))
        chosen["hc"][f] = caruana(obj, len(NAMES))[0]
        chosen["nm"][f] = nelder_mead(obj, len(NAMES))[0]
        print(f" fold {f}: hc w={np.round(chosen['hc'][f] / chosen['hc'][f].sum(), 3)}  nm w={np.round(chosen['nm'][f] / chosen['nm'][f].sum(), 3)}", flush=True)
    for meth in ("hc", "nm"):
        tot = _add_all(counts(chosen[meth][f])[f] for f in folds)
        ch = [weighted(counts(chosen[meth][f])[f]) - weighted(base[f]) for f in folds]
        print(f"[{meth}] honest {weighted(tot):.4f} vs deployed {weighted(_add_all(base.values())):.4f} "
              f"({weighted(tot) - weighted(_add_all(base.values())):+.4f}) | per-fold change " + " ".join(f"{c:+.4f}" for c in ch)
              + "  " + " ".join(f"{c[:8]} {tuple(int(v) for v in tot[c])}" for c in K if c in tot), flush=True)
    full = {"hc": caruana(lambda w: weighted(_add_all(counts(w).values())), len(NAMES))[0]}
    print(f"all-fold hill-climbing weights (would be deployed): {dict(zip(NAMES, np.round(full['hc'] / full['hc'].sum(), 3)))}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    wsel = lambda st, r: eq if st == "eq" else chosen[st][RF[r]]
    CS = site_counts(lambda st, r: fc(pipeline(wsel(st, r), [r])), ("eq", "hc", "nm"), regions, gtr)
    for meth in ("hc", "nm"):
        report(f"member weights {meth} (per-fold, honest)", CS["eq"], CS[meth])


if __name__ == "__main__":
    self_test()
    if sys.argv[1:2] != ["--self-test"]:
        main()
