"""Fixed satellite coordinate-consensus audit on geographic OOF.

Keeps deployed clusters, classes and scores exactly; changes only the point
within each cluster. This tests the useful part of WBF without substituting a
box-IoU algorithm for the deployed 8 m point fusion.
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import _add_all, fold_counts, weighted
from koz3.sp_fuse import fam, fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry

K = ("kurgany_povrezhdennye", "kurgany_tselye")
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")


def coordinate_variants(frames, match_m=8.0):
    allc = pd.concat([m.assign(mi=i) for i, m in enumerate(frames)], ignore_index=True)
    allc["rank"] = 0.0
    for (_, _), idx in allc.groupby(["region", "mi"]).groups.items():
        allc.loc[idx, "rank"] = rankdata(allc.loc[idx, "score"].to_numpy(), method="average") / len(idx)
    allc["fam"] = allc.cls.map(fam)
    rows = []
    for (region, family), g in allc.groupby(["region", "fam"]):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        xy = g[["x", "y"]].to_numpy()
        scores = g.score.to_numpy()
        ranks = g["rank"].to_numpy()
        model_ids = g.mi.to_numpy()
        classes = g.cls.to_numpy()
        tree = cKDTree(xy)
        lat = np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)
        scale = 1 / np.cos(np.radians(lat))
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            idx = np.array([j for j in tree.query_ball_point(xy[i], match_m * scale) if not used[j]], dtype=int)
            used[idx] = True
            best = np.zeros(len(frames), dtype=float)
            rank_best = np.zeros(len(frames), dtype=float)
            for j in idx:
                best[model_ids[j]] = max(best[model_ids[j]], scores[j])
                rank_best[model_ids[j]] = max(rank_best[model_ids[j]], ranks[j])
            score = float(best.mean())
            rank_score = float(rank_best.mean())
            skp = float(max(scores[j] for j in idx if classes[j] == K[0])) if any(classes[idx] == K[0]) else 0.0
            skt = float(max(scores[j] for j in idx if classes[j] == K[1])) if any(classes[idx] == K[1]) else 0.0
            # One strongest detection per model avoids a model with many boxes
            # dominating the location. All variants retain the deployed anchor.
            winners = np.array([max((j for j in idx if model_ids[j] == m), key=lambda j: scores[j])
                                for m in np.unique(model_ids[idx])], dtype=int)
            w = scores[winners]
            consensus = np.average(xy[winners], axis=0, weights=w)
            rows.append((region, classes[i], score, rank_score, xy[i, 0], xy[i, 1],
                         consensus[0], consensus[1], skp, skt))
    return pd.DataFrame(rows, columns=["region", "cls", "score", "rank_score", "x", "y", "cx", "cy", "s_kp", "s_kt"])


def main():
    M = models(extra=[n for n in NAMES if n not in ("dfine", "rtdetr", "hm")])
    regions = sorted(set.intersection(*[set(M[n].region) for n in NAMES]))
    frames = [M[n][M[n].region.isin(regions)] for n in NAMES]
    F = coordinate_variants(frames)
    baseline = fuse(frames, np.ones(len(frames)))
    assert len(F) == len(baseline)
    assert (F[["region", "cls"]].to_numpy() == baseline[["region", "cls"]].to_numpy()).all()
    assert np.allclose(F[["score", "x", "y", "s_kp", "s_kt"]].to_numpy(dtype=float),
                       baseline[["score", "x", "y", "s_kp", "s_kt"]].to_numpy(dtype=float))
    print("fusion parity verified", len(F), flush=True)
    F = F[F.score >= np.where(F.cls == "arkhitektury", .15, .25)].reset_index(drop=True)
    geometry = scene_geometry()
    keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - x, r.y - y) / max(h, 1), sm) for x, y, h, sm in geometry.get(r.region, [])]
                or [(0., True)])
        keep.append(d[1] or d[0] <= .5 or r.score >= .35)
    F = F[np.array(keep)].reset_index(drop=True)
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {fold: to_gt_geojson(L[L.region.map(RF) == fold]) for fold in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet"))
    S = S[S.region.isin(regions)]

    def make(mode, fold):
        g = F[F.region.map(RF) == fold].copy()
        if mode == "consensus":
            g[["x", "y"]] = g[["cx", "cy"]].to_numpy()
        elif mode != "anchor":
            raise ValueError(mode)
        extra = []
        for r in g[g.cls.isin(K)].itertuples():
            other = K[1] if r.cls == K[0] else K[0]
            if (r.s_kt if other == K[1] else r.s_kp) >= .4:
                extra.append((r.region, other, r.score * .99, r.x, r.y))
        g = g[["region", "cls", "score", "x", "y"]]
        if extra:
            g = pd.concat([g, pd.DataFrame(extra, columns=g.columns)], ignore_index=True)
        return fc(gate(g, S[S.region.map(RF) == fold]))

    C = fold_counts(make, ["anchor", "consensus"], folds, GT)
    for mode in ("anchor", "consensus"):
        total = _add_all(C[mode].values())
        print(mode, "pooled", weighted(total), "folds", {f: weighted(C[mode][f]) for f in folds}, flush=True)
        print(mode, "classes", {k: tuple(map(int, v)) for k, v in total.items()}, flush=True)


if __name__ == "__main__":
    main()
