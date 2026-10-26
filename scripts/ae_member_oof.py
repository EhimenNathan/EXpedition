"""Ae kurgans from a single strong member (or max-fusion of FCOS + joint D-FINE) + a local-density filter, judged on the
VALIDATION-SHAPED pooled metric (scripts/ae_cluster_oof.py protocol). Data: per member, cemetery sheets reach 62–90 %
hits at score ≥ 0.2 (FCOS Тимерево 193 dets, 90 %), open-field sheets 1–11 % → keep a detection only when ≥ m other
detections ≥ t lie within R metres (a carpet of mounds), which is computable at test time.
  python scripts/ae_member_oof.py
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
SRC = {"fcos": "work/ae/oof_ae_fcos_cands.parquet", "dfj": "work/ae/oof_ae_dfj_cands.parquet"}


def load(member):
    if member == "max":                 # max-fusion within 6 m: keep the higher-scoring of FCOS / D-FINE
        from scipy.spatial import cKDTree
        a, b = load("fcos"), load("dfj"); out = []
        for reg in sorted(set(a.region) | set(b.region)):
            g = pd.concat([a[a.region == reg], b[b.region == reg]]).sort_values("score", ascending=False).reset_index(drop=True)
            if not len(g):
                continue
            k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
            t = cKDTree(g[["x", "y"]].values); used = np.zeros(len(g), bool)
            for i in range(len(g)):
                if used[i]:
                    continue
                idx = [j for j in t.query_ball_point(g.loc[i, ["x", "y"]].values, 6 * k) if not used[j]]; used[idx] = True
                out.append(g.loc[i])
        return pd.DataFrame(out).reset_index(drop=True)
    d = pd.read_parquet(os.path.join(ROOT, SRC[member]))
    return d[d.cls.isin(K) & (d.score >= 0.15)][["region", "cls", "score", "x", "y"]].reset_index(drop=True)


def main():
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    from val_sim import pts
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    ae_regions = sorted(set(L[L["mod"] == "Ae"].region))
    GT = {r: to_gt_geojson(L[(L["mod"] == "Ae") & (L.region == r)]) for r in ae_regions}
    C = pickle.load(open(os.path.join(ROOT, "work/joint_thr_counts.pkl"), "rb"))
    C = {k: v for k, v in C.items() if k[0] != "Ae"}
    settings = [None]
    for member in ("fcos", "dfj", "max"):
        D = load(member); print(member, len(D), D.region.nunique(), flush=True)
        dens = {}
        for t, R in itertools.product((0.2, 0.3, 0.4), (60, 120)):
            col = np.zeros(len(D), int)
            for reg, g in D.groupby("region"):
                k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
                s = g[g.score >= t]
                if not len(s):
                    continue
                tr = cKDTree(s[["x", "y"]].values)
                col[D.index.get_indexer(g.index)] = [len(q) - (sc >= t) for q, sc in zip(tr.query_ball_point(g[["x", "y"]].values, R * k), g.score)]
            dens[(t, R)] = col
        for t, R, m in itertools.product((0.2, 0.3, 0.4), (60, 120), (0, 5, 15, 30)):
            if m == 0 and R == 120:
                continue
            st = (member, t, R, m); settings.append(st)
            keep = (D.score.values >= t) & (dens[(t, R)] >= m)
            for r in ae_regions:
                g = D[keep & (D.region == r).values]
                _, res = score({"type": "FeatureCollection", "features": pts(g, "ae")}, GT[r])
                C[("Ae", r, st)] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
    for r in ae_regions:
        _, res = score({"type": "FeatureCollection", "features": []}, GT[r]); C[("Ae", r, None)] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
    pickle.dump(C, open(os.path.join(ROOT, "work/ae_member_counts.pkl"), "wb"))
    tot = {st: _add_all(C[("Ae", r, st)] for r in ae_regions) for st in settings}
    share = lambda st: (sum(tot[st][c][0] for c in K), sum(tot[st][c][1] for c in K))
    for st in sorted(settings[1:], key=lambda s: -share(s)[0] / max(sum(share(s)), 1))[:12]:
        tp, fp = share(st); print(f"{st}: kurgan TP {tp} FP {fp} share {tp / max(tp + fp, 1):.3f}")
    regs = {m: sorted({k[1] for k in C if k[0] == m}) for m in COMP}
    dep = {"SpOr": 1.0, "Li": 0.15, "Or": 0.45}
    def draws(pool, n, rng):
        return [{m: list(rng.choice(pool[m], COMP[m], replace=len(pool[m]) < COMP[m])) for m in COMP} for _ in range(n)]
    def ev(dr, st):
        return np.mean([weighted(_add_all(C[(m, r, st if m == "Ae" else dep.get(m))] for m in COMP for r in d[m])) for d in dr])
    rng = np.random.default_rng(0); full = draws(regs, 300, rng)
    top = sorted(settings[1:], key=lambda s: -ev(full, s))[:8]
    print(f"pooled draws: Ae off {ev(full, None):.4f} | best: " + ", ".join(f"{s}:{ev(full, s):.4f}" for s in top))
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
    print("chosen:\n" + pd.Series([str(c) for c in ch]).value_counts().head(6).to_string())


if __name__ == "__main__":
    main()
