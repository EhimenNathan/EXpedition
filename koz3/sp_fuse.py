"""Optical detector fusion (OOF study and inference share this): per region and class family, candidates of all
models within match_m metres are one object; score = weighted mean of each model's best score there (missing = 0);
class = highest-scoring member's class. Honest OOF: D-FINE + RT-DETRv2 + heatmap, equal weights, threshold 0.25 →
0.2729 (nested over 4 site folds; the same threshold was chosen in every fold)."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K = ("kurgany_povrezhdennye", "kurgany_tselye")
fam = lambda c: "k" if c in K else c


def fuse(models, weights, match_m=8.0):
    """models: list of DataFrames (region, cls, score, x, y). → fused DataFrame."""
    allc = pd.concat([m.assign(mi=i) for i, m in enumerate(models)], ignore_index=True)
    allc["fam"] = allc.cls.map(fam)
    out = []
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
            s = float(np.dot(weights, best) / np.sum(weights))
            skp = max([g.score[j] for j in idx if g.cls[j] == K[0]], default=0.0)    # best member score per kurgan
            skt = max([g.score[j] for j in idx if g.cls[j] == K[1]], default=0.0)    # class (damaged / intact hedge)
            out.append((reg, g.cls[i], s, g.x[i], g.y[i], float(skp), float(skt)))
    return pd.DataFrame(out, columns=["region", "cls", "score", "x", "y", "s_kp", "s_kt"])
