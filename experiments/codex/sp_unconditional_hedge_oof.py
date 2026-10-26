"""Satellite damaged/intact hedge from detector disagreement (nested): in each fused kurgan object, if the members' best
score for the OTHER kurgan class is ≥ tau, that class is emitted as well (same position). On the v12 pipeline (5-member
fusion, global 0.25 / architecture 0.15, framing prior x 0.5 / t2 0.35, hillfort gate)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds  # noqa: E402
from sp_fusion_oof import fc  # noqa: E402
from sp_fusion4_oof import gate  # noqa: E402
from sp_rescore_oof import scene_geometry  # noqa: E402

K = ("kurgany_povrezhdennye", "kurgany_tselye")
fam = lambda c: "k" if c in K else c


def fuse_cls(models, match_m=8.0):
    """koz3.sp_fuse.fuse + per-cluster best member score for each kurgan class."""
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
            out.append((reg, g.cls[i], float(best.mean()), g.x[i], g.y[i], sc[K[0]], sc[K[1]]))
    return pd.DataFrame(out, columns=["region", "cls", "score", "x", "y", "s_kp", "s_kt"])


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    names = tuple(os.environ.get("SP_NAMES", "dfine,rtdetr,hm,dfnodn,balq,balq3").split(","))
    M = models(extra=[n for n in names if n not in ("dfine", "rtdetr", "hm")]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    F = fuse_cls([M[n] for n in names])
    F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True)
    geom = scene_geometry(); keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
    F = F[np.array(keep)].reset_index(drop=True)

    def make(s, f):
        tau = s[0]
        G = F[F.region.map(RF) == f]
        extra = []
        if tau < 9:
            for r in G[G.cls.isin(K)].itertuples():
                other = K[1] if r.cls == K[0] else K[0]
                so = r.s_kt if other == K[1] else r.s_kp
                if so >= tau:
                    extra.append((r.region, other, r.score * 0.99, r.x, r.y))
        G = G[["region", "cls", "score", "x", "y"]]
        if extra:
            G = pd.concat([G, pd.DataFrame(extra, columns=G.columns)], ignore_index=True)
        return fc(gate(G, S[S.region.map(RF) == f]))
    # Fixed comparison: deployed disagreement hedge versus emitting both classes.
    # No threshold search or held-out selection.
    SS = [(0.4,), (0.0,)]
    C = fold_counts(make, SS, folds, gt)
    for s in SS:
        tot = _add_all(C[s].values())
        print(f"tau {s[0]}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot['kurgany_povrezhdennye'])}  kt {tuple(int(v) for v in tot['kurgany_tselye'])}")
    base = weighted(_add_all(C[(0.4,)].values()))
    both = weighted(_add_all(C[(0.0,)].values()))
    print(f"deployed conditional {base:.4f} | unconditional {both:.4f} | delta {both-base:+.4f}")
    for f in folds:
        a = weighted(C[(0.4,)][f]); b = weighted(C[(0.0,)][f])
        print(f"fold {f}: conditional {a:.4f} unconditional {b:.4f} delta {b-a:+.4f}")


if __name__ == "__main__":
    main()
