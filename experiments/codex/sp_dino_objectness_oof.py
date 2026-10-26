"""Precommitted DINO candidate filter, trained on 0-2 and tested on fold 3.

Reuses the exact fusion and scorer in sp_morph_objectness_oof. Only candidates
covered by the frozen small-scene token cache can change. The threshold is
selected on folds 0-2 before fold 3 is scored.
"""
import json
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import weighted, _add_all
from koz3.score import score
from sp_assemble_oof import models, region_folds
from sp_dual_oof import fuse_cls
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry

HERE = Path(__file__).resolve().parent
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
THRESHOLDS = (.1, .15, .2, .25, .3, .35, .4, .5, .6, .7)

def main():
    P = pd.read_parquet(HERE / "sp_dino_candidate_probe.parquet")
    P = P[["region", "cls", "score", "x", "y", "p_dino"]]
    M = models(extra=list(NAMES[3:]))
    regions = sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
    RF = region_folds()
    F = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES])
    F = F[(F.cls.isin(K) & (F.score >= .08)) |
          ((F.cls == "arkhitektury") & (F.score >= .15)) |
          ((~F.cls.isin(K + ("arkhitektury",))) & (F.score >= .25))].copy()
    F = F.merge(P, how="left", on=["region", "cls", "score", "x", "y"], validate="one_to_one")
    geom = scene_geometry()
    keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])]
                or [(0.0, True)])
        keep.append(d[1] or d[0] <= .5 or r.score >= .35)
    F = F[np.array(keep)]
    S = pd.read_parquet(ROOT / "work/spor/oof_scene_cls.parquet")
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in range(4)}
    print("DINO coverage", int(F.p_dino.notna().sum()), "of", len(F), flush=True)

    def make(fold, threshold=None):
        g = F[F.region.map(RF) == fold].copy()
        if threshold is None:
            mask = (~g.cls.isin(K)) | (g.score >= .25)
        else:
            mask = (~g.cls.isin(K)) | (g.p_dino >= threshold) | (g.p_dino.isna() & (g.score >= .25))
        g = g[mask].copy()
        if threshold is not None:
            q = g.cls.isin(K) & g.p_dino.notna()
            g.loc[q, "score"] = g.loc[q, "p_dino"]
        extra = []
        for r in g[g.cls.isin(K)].itertuples():
            other = K[1] if r.cls == K[0] else K[0]
            if (r.s_kt if other == K[1] else r.s_kp) >= .4:
                extra.append((r.region, other, r.score * .99, r.x, r.y))
        g = g[["region", "cls", "score", "x", "y"]]
        if extra:
            g = pd.concat([g, pd.DataFrame(extra, columns=g.columns)], ignore_index=True)
        return fc(gate(g, S[S.region.map(RF) == fold]))

    def scored_counts(pred, target):
        return {c: (v["tp"], v["fp"], v["fn"]) for c, v in score(pred, target)[1].items()}

    baseline = {f: scored_counts(make(f), gt[f]) for f in range(4)}
    C = {t: {f: scored_counts(make(f, t), gt[f]) for f in range(4)} for t in THRESHOLDS}
    chosen = max(THRESHOLDS, key=lambda t: weighted(_add_all(C[t][f] for f in (0, 1, 2))))
    result = {
        "chosen_threshold_on_folds012": chosen,
        "base_inner": weighted(_add_all(baseline[f] for f in (0, 1, 2))),
        "dino_inner": weighted(_add_all(C[chosen][f] for f in (0, 1, 2))),
        "base_fold3": weighted(baseline[3]),
        "dino_fold3": weighted(C[chosen][3]),
        "base_all": weighted(_add_all(baseline.values())),
        "dino_all": weighted(_add_all(C[chosen].values())),
    }
    print(json.dumps(result, indent=2), flush=True)
    (HERE / "sp_dino_objectness_verdict.json").write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
