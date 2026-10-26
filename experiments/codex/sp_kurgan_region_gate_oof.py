"""Nested geographic OOF study of a region-level satellite kurgan gate."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import _add_all, fold_counts, nested, weighted
from sp_assemble_oof import region_folds
from sp_fusion_oof import fc


def main():
    F = pd.read_parquet(Path(__file__).with_name("sp_agreement_oof.parquet"))
    regions = sorted(set(F.region))
    F = F[F.score >= .25].copy()
    k = F.cls.isin(("kurgany_povrezhdennye", "kurgany_tselye"))
    groups = F[k].groupby("region").score.agg(["size", "max"])
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    base = (1, .0)
    settings = [base] + [(n, m) for n in (2, 3, 4, 5, 7, 10) for m in (.0, .3, .4, .5) if (n, m) != base]

    def make(setting, fold):
        n, mx = setting
        z = F[F.region.map(RF) == fold]
        keep = set(groups[(groups["size"] >= n) & (groups["max"] >= mx)].index)
        z = z[~z.cls.isin(("kurgany_povrezhdennye", "kurgany_tselye")) | z.region.isin(keep)]
        return fc(z)

    C = fold_counts(make, settings, folds, GT)
    honest, chosen, oracle, oracle_setting, _ = nested(C)
    summary = {"regions": len(regions), "baseline": weighted(_add_all(C[base].values())), "nested": honest,
               "chosen_by_fold": {str(f): list(s) for f, s in chosen.items()},
               "oracle": oracle, "oracle_setting": list(oracle_setting),
               "fixed": {str(s): {"pooled": weighted(_add_all(C[s].values())),
                                  "folds": {str(f): weighted(C[s][f]) for f in folds}}
                         for s in (base, (2,.0), (3,.0), (5,.0), (3,.4))}}
    Path(__file__).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
