"""Geographic nested OOF audit of agreement-aware optical fusion thresholds.

Uses the same greedy 8 m clusters as the deployed satellite fusion. Scores
clusters by the deployed mean score, but keeps their model-support count so
single-model and multi-model clusters can have separate thresholds.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import _add_all, fold_counts, nested, weighted
from koz3.sp_fuse import fam, fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc


def fuse_with_agreement(model_frames, match_m=8.0):
    allc = pd.concat([m.assign(mi=i) for i, m in enumerate(model_frames)], ignore_index=True)
    allc["fam"] = allc.cls.map(fam)
    out = []
    for (reg, family), g in allc.groupby(["region", "fam"]):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        xy = g[["x", "y"]].to_numpy()
        tree = cKDTree(xy)
        scale = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            idx = [j for j in tree.query_ball_point(xy[i], match_m * scale) if not used[j]]
            used[idx] = True
            best = np.zeros(len(model_frames))
            for j in idx:
                best[g.mi[j]] = max(best[g.mi[j]], g.score[j])
            out.append((reg, g.cls[i], best.mean(), g.x[i], g.y[i], int((best > 0).sum()), *best))
    return pd.DataFrame(out, columns=["region", "cls", "score", "x", "y", "support"] + [f"m{i}" for i in range(len(model_frames))])


def main():
    M = models()
    regions = sorted(set.intersection(*(set(m.region) for m in M.values())))
    M = {k: v[v.region.isin(regions)] for k, v in M.items()}
    F = fuse_with_agreement(list(M.values()))
    F.to_parquet(Path(__file__).with_name("sp_agreement_oof.parquet"), index=False)
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    base = (0.25, 0.25)
    settings = [base] + [(a, b) for a in (.25, .30, .35, .40, .45, .50) for b in (.10, .15, .20, .25, .30) if (a, b) != base]

    def make(s, fold):
        one, multiple = s
        g = F[F.region.map(RF) == fold]
        g = g[((g.support == 1) & (g.score >= one)) | ((g.support >= 2) & (g.score >= multiple))]
        return fc(g)

    C = fold_counts(make, settings, folds, GT)
    honest, chosen, oracle, oracle_s, _ = nested(C)
    baseline = weighted(_add_all(C[base].values()))
    full = fuse(list(M.values()), np.ones(len(M)))
    check = pd.concat([F[["region", "cls", "score", "x", "y"]].reset_index(drop=True), full.reset_index(drop=True)], axis=1)
    # The row order and the core fields must match the deployment fusion.
    assert len(F) == len(full) and (F[["region", "cls"]].to_numpy() == full[["region", "cls"]].to_numpy()).all()
    assert np.allclose(F[["score", "x", "y"]].to_numpy(), full[["score", "x", "y"]].to_numpy())
    result = {"regions": len(regions), "candidates": len(F), "support_counts": F.support.value_counts().sort_index().to_dict(),
              "baseline": baseline, "nested": honest, "chosen_by_fold": {str(f): list(s) for f, s in chosen.items()},
              "oracle": oracle, "oracle_setting": list(oracle_s),
              "fixed": {str(s): {"pooled": weighted(_add_all(C[s].values())), "folds": {str(f): weighted(C[s][f]) for f in folds}}
                        for s in (base, (.35, .15), (.40, .20))}}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
