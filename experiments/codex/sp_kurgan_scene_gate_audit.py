"""Official-metric audit of frozen-image kurgan scene gate using geographic OOF probabilities."""
import json
import sys
from pathlib import Path

import geopandas as gpd
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
    P = pd.read_parquet(Path(__file__).with_name("sp_kurgan_scene_oof.parquet")).set_index("region").p.to_dict()
    F["scene_p"] = F.region.map(P)
    assert F.scene_p.notna().all()
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    settings = [None, .05, .1, .15, .2, .25, .3, .4, .5, .6]

    def make(t, fold):
        z = F[F.region.map(RF) == fold]
        if t is not None:
            z = z[~z.cls.isin(("kurgany_povrezhdennye", "kurgany_tselye")) | (z.scene_p >= t)]
        return fc(z)

    C = fold_counts(make, settings, folds, GT)
    honest, chosen, oracle, oracle_t, _ = nested(C)
    out = {"regions": len(regions), "baseline": weighted(_add_all(C[None].values())), "nested": honest,
           "chosen_by_fold": {str(f): t for f, t in chosen.items()}, "oracle": oracle,
           "oracle_threshold": oracle_t,
           "fixed": {str(t): {"pooled": weighted(_add_all(C[t].values())),
                              "folds": {str(f): weighted(C[t][f]) for f in folds},
                              "counts": {k: list(map(int, v)) for k, v in _add_all(C[t].values()).items()}}
                    for t in settings}}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
