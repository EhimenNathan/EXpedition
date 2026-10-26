"""Nested geographic test of per-site/per-member percentile score fusion.

Clusters, class, coordinates, raw confidence hedge and framing decision stay
deployed. Only the fused score used for the detection threshold is replaced.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from sp_coord_consensus_oof import coordinate_variants, NAMES, ROOT, K
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry
from koz3.labels import to_gt_geojson
from koz3.nested import _add_all, fold_counts, nested, weighted


def main():
    M = models(extra=[n for n in NAMES if n not in ("dfine", "rtdetr", "hm")])
    regions = sorted(set.intersection(*[set(M[n].region) for n in NAMES]))
    F = coordinate_variants([M[n][M[n].region.isin(regions)] for n in NAMES])
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet"))
    S = S[S.region.isin(regions)]
    geom = scene_geometry()
    keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])]
                or [(0.0, True)])
        keep.append(d[1] or d[0] <= 0.5 or r.score >= .35)
    F = F[np.array(keep)].reset_index(drop=True)
    settings = [("raw", .25)] + [("rank", t) for t in (.1, .15, .2, .25, .3, .35, .4, .45,
                                                 .5, .55, .6, .65, .7, .75, .8)]

    def make(setting, fold):
        mode, threshold = setting
        g = F[F.region.map(RF) == fold]
        v = g.score if mode == "raw" else g.rank_score
        g = g[v >= np.where(g.cls == "arkhitektury", threshold * .6, threshold)].copy()
        extra = []
        for r in g[g.cls.isin(K)].itertuples():
            other = K[1] if r.cls == K[0] else K[0]
            if (r.s_kt if other == K[1] else r.s_kp) >= .4:
                extra.append((r.region, other, r.score * .99, r.x, r.y))
        g = g[["region", "cls", "score", "x", "y"]]
        if extra:
            g = pd.concat([g, pd.DataFrame(extra, columns=g.columns)], ignore_index=True)
        return fc(gate(g, S[S.region.map(RF) == fold]))

    C = fold_counts(make, settings, folds, GT)
    base = weighted(_add_all(C[("raw", .25)].values()))
    assert abs(base - .36665956497943597) < 1e-12, base
    rank = {k: v for k, v in C.items() if k[0] == "rank"}
    honest, chosen, oracle, oracle_setting, total = nested(rank)
    print("raw baseline", base, flush=True)
    print("rank nested", honest, "chosen", chosen, "oracle", oracle, oracle_setting, flush=True)
    for setting in rank:
        print(setting, weighted(_add_all(rank[setting].values())),
              {f: weighted(rank[setting][f]) for f in folds}, flush=True)
    print("nested classes", {k: tuple(map(int, v)) for k, v in total.items()}, flush=True)


if __name__ == "__main__":
    main()
