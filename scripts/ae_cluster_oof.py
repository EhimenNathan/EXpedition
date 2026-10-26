"""Ae (German film) kurgans: member fusion + cemetery-cluster prior, judged by its effect on the VALIDATION-SHAPED
pooled metric (public composition: Ae ≈ 525–600 GT, the largest polygon set; v9 Ae P 0.045).

Members (OOF over the Ae region folds): satellite D-FINE on film (spdet), joint D-FINE (ae_dfj), FCOS (ae_fcos),
heatmap (hm). Kurgan candidates fused within 8 m (score = mean over members of the best member score; class = best
class; s_kp / s_kt kept). Cluster prior: score × (0.5 + 0.5·clip(n/K, 0, 2)), n = fused candidates with score ≥ snb
within R. Emission: primary class (+ the other kurgan class if its best member score ≥ 0.4).
Per region and setting: per-class counts vs the region's Ae GT; then joint_thr's protocol — validation-shaped draws
(30 SpOr, 13 Li, 4 Ae, 2 MagGr, 1 EtGr, 1 TerFl, 1 Or) with everything else at deployed settings, the Ae setting chosen
on half of the regions and scored on the other half (10 splits).
  python scripts/ae_cluster_oof.py
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
K = ("kurgany_povrezhdennye", "kurgany_tselye")
COMP = {"SpOr": 30, "Li": 13, "Ae": 4, "MagGr": 2, "EtGr": 1, "TerFl": 1, "Or": 1}


def fuse(cands, n_members, match_m=8.0):
    from scipy.spatial import cKDTree
    out = []
    for reg, g in cands.groupby("region"):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
        t = cKDTree(g[["x", "y"]].values); used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            idx = [j for j in t.query_ball_point(g.loc[i, ["x", "y"]].values, match_m * k) if not used[j]]
            used[idx] = True
            best = np.zeros(n_members)
            for j in idx:
                best[g.mi[j]] = max(best[g.mi[j]], g.score[j])
            skp = max([g.score[j] for j in idx if g.cls[j] == K[0]], default=0.0)
            skt = max([g.score[j] for j in idx if g.cls[j] == K[1]], default=0.0)
            out.append((reg, g.cls[i], best.mean(), int((best > 0).sum()), g.x[i], g.y[i], skp, skt))
    return pd.DataFrame(out, columns=["region", "cls", "score", "nm", "x", "y", "s_kp", "s_kt"])


def main():
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    from val_sim import pts
    srcs = {"spdet": "work/ae/oof_spdet_cands.parquet", "dfj": "work/ae/oof_ae_dfj_cands.parquet",
            "fcos": "work/ae/oof_ae_fcos_cands.parquet", "hm": "work/ae/oof_hm_cands.parquet"}
    M = []
    for i, (nm, f) in enumerate(srcs.items()):
        d = pd.read_parquet(os.path.join(ROOT, f)); d = d[d.cls.isin(K) & (d.score >= 0.1)][["region", "cls", "score", "x", "y"]]
        print(nm, len(d), d.region.nunique(), flush=True); M.append(d.assign(mi=i))
    regions = sorted(set.intersection(*[set(m.region) for m in M]))
    F = fuse(pd.concat([m[m.region.isin(regions)] for m in M], ignore_index=True), len(M))
    print("fused", len(F), "regions", len(regions), flush=True)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    GT = {r: to_gt_geojson(L[(L["mod"] == "Ae") & (L.region == r)]) for r in regions}
    nb = {}
    for (R, snb) in itertools.product((100, 250), (0.2, 0.3)):
        col = np.zeros(len(F))
        for reg, g in F.groupby("region"):
            k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
            s = g[g.score >= snb]; t = cKDTree(s[["x", "y"]].values) if len(s) else None
            col[F.index.get_indexer(g.index)] = [len(t.query_ball_point(p, R * k)) - (sc >= snb) if t is not None else 0
                                                 for p, sc in zip(g[["x", "y"]].values, g.score)]
        nb[(R, snb)] = col
    settings = [None] + [(t, None) for t in (0.3, 0.4, 0.5, 0.6)] + \
               [(t, (R, Kc, snb)) for t in (0.3, 0.4, 0.5, 0.6) for R, snb in nb for Kc in (3, 8)]
    C = pickle.load(open(os.path.join(ROOT, "work/joint_thr_counts.pkl"), "rb"))
    C = {k: v for k, v in C.items() if k[0] != "Ae"}
    for st in settings:
        for r in regions:
            if st is None:
                g = F.iloc[:0]
            else:
                t, pr = st; sc = F.score.values.copy()
                if pr is not None:
                    R, Kc, snb = pr; sc = sc * (0.5 + 0.5 * np.clip(nb[(R, snb)] / Kc, 0, 2))
                g = F[(F.region == r) & (sc >= t)].assign(score=sc[(F.region == r).values & (sc >= t)])
            ex = [(r_.region, K[1] if r_.cls == K[0] else K[0], r_.score * 0.99, r_.x, r_.y) for r_ in g.itertuples()
                  if (r_.s_kt if r_.cls == K[0] else r_.s_kp) >= 0.4]
            G = pd.concat([g[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
            _, res = score({"type": "FeatureCollection", "features": pts(G, "ae")}, GT[r])
            C[("Ae", r, st)] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
    tot = {st: _add_all(C[("Ae", r, st)] for r in regions) for st in settings}
    for st in settings[:5] + sorted(settings[5:], key=lambda s: -weighted(tot[s]))[:6]:
        tp = sum(tot[st][c][0] for c in K); fp = sum(tot[st][c][1] for c in K); fn = sum(tot[st][c][2] for c in K)
        print(f"Ae {st}: kurgan TP {tp} FP {fp} FN {fn}  TP share {tp / max(tp + fp, 1):.3f}", flush=True)
    pickle.dump(C, open(os.path.join(ROOT, "work/ae_cluster_counts.pkl"), "wb"))
    regs = {m: sorted({k[1] for k in C if k[0] == m}) for m in COMP}
    dep = {"SpOr": 1.0, "Li": 0.15, "Or": 0.45}
    def draws(pool, n, rng):
        return [{m: list(rng.choice(pool[m], COMP[m], replace=len(pool[m]) < COMP[m])) for m in COMP} for _ in range(n)]
    def ev(dr, st):
        return np.mean([weighted(_add_all(C[(m, r, st if m == "Ae" else dep.get(m))] for m in COMP for r in d[m])) for d in dr])
    rng = np.random.default_rng(0); full = draws(regs, 300, rng)
    print(f"pooled draws: Ae off {ev(full, None):.4f} | " + " ".join(f"{s}:{ev(full, s):.4f}" for s in settings[1:5]))
    print("best prior settings:", ", ".join(f"{s}:{ev(full, s):.4f}" for s in sorted(settings[5:], key=lambda s: -ev(full, s))[:4]))
    hon, base, ch = [], [], []
    for s in range(10):
        rs = np.random.default_rng(100 + s); A_, B_ = {}, {}
        for m, v in regs.items():
            p = list(rs.permutation(v)); h = max(1, len(p) // 2); A_[m], B_[m] = (p[:h], p[h:] or p[:h]) if len(p) > 1 else (p, p)
        for tr, te in ((A_, B_), (B_, A_)):
            dtr, dte = draws(tr, 120, rs), draws(te, 120, rs)
            best = max(settings, key=lambda st: ev(dtr, st)); hon.append(ev(dte, best)); base.append(ev(dte, None)); ch.append(best)
    g = np.array(hon) - np.array(base)
    print(f"HONEST Ae-on vs off: {np.mean(hon):.4f} vs {np.mean(base):.4f}  gain {g.mean():+.4f}  wins {int((g > 0).sum())}/{len(g)}")
    print("chosen:", pd.Series([str(c) for c in ch]).value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
