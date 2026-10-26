"""Nested OOF audit of selective intact/damaged dual emission in satellite fusion."""
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
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc

K0, K1 = "kurgany_povrezhdennye", "kurgany_tselye"


def main():
    M = models()
    regions = sorted(set.intersection(*(set(v.region) for v in M.values())))
    M = {k: v[v.region.isin(regions)] for k, v in M.items()}
    F = fuse(list(M.values()), np.ones(3))
    F = F[F.score >= .25].copy()
    RF = region_folds()
    folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    settings = [("none", 1., 0.)]
    for direction in ("both", "damage_to_intact", "intact_to_damage"):
        for threshold in (.35, .45, .55, .65, .75, .85):
            for factor in (.05, .25, .5):
                settings.append((direction, threshold, factor))

    def make(setting, fold):
        direction, threshold, factor = setting
        Z = F[F.region.map(RF) == fold]
        if direction == "none":
            return fc(Z)
        allowed = [K0, K1] if direction == "both" else ([K0] if direction == "damage_to_intact" else [K1])
        D = Z[(Z.cls.isin(allowed)) & (Z.score >= threshold)].copy()
        D["cls"] = D.cls.map({K0: K1, K1: K0})
        D["score"] *= factor
        return fc(pd.concat([Z, D], ignore_index=True))

    C = fold_counts(make, settings, folds, GT)
    h, chosen, o, oracle, by_cls = nested(C)
    baseline = nested({settings[0]: C[settings[0]]})[0]
    fixed = {}
    for setting in (settings[0], ("intact_to_damage", .35, .05), ("both", .45, .05)):
        fixed[str(setting)] = {
            "pooled": weighted(_add_all(C[setting].values())),
            "fold_scores": {str(f): weighted(C[setting][f]) for f in folds},
            "class_counts": {k: list(map(int, v)) for k, v in _add_all(C[setting].values()).items()},
        }
    out = {"baseline": baseline, "nested": h, "selected_by_fold": {str(k): v for k, v in chosen.items()},
           "oracle": o, "oracle_setting": oracle, "nested_class_counts": {k: list(map(int, v)) for k,v in by_cls.items()},
           "fixed": fixed}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
