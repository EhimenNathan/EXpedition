"""Evaluate completed private Kaggle 0.3 m D-FINE OOF beside 0.5 m experts.

No model is trained here. Every score uses held-out geographic site predictions,
the same common sites, and the official scorer. Thresholds are chosen on the
other three folds for each held-out fold.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import fold_counts, nested
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc, load_theirs, to3857


def main():
    crs = pd.read_csv(ROOT / "kaggle_import/code/docs/site_crs.csv").set_index("site")
    crs_of = {s: f"EPSG:{int(e)}" for s, e in crs["json_epsg"].items()}
    q = ROOT / "Codex Work/kaggle_sp03_reference/sp_runs/sp03_dfine_l"
    S = to3857(load_theirs(q), crs_of)
    original = models()
    common = set(S.region).intersection(*(set(v.region) for v in original.values()))
    M = {k: v[v.region.isin(common)] for k, v in original.items()}
    S = S[S.region.isin(common)]
    RF = region_folds()
    folds = sorted({RF[r] for r in common})
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(common)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    print("common sites", len(common), "GT", len(L), "sp03 candidates", len(S), flush=True)

    variants = {
        "base": (list(M.values()), np.ones(3)),
        "sp03_only": ([S], np.ones(1)),
        "add_sp03_025": (list(M.values()) + [S], np.array([1, 1, 1, .25])),
        "add_sp03_050": (list(M.values()) + [S], np.array([1, 1, 1, .5])),
        "add_sp03_100": (list(M.values()) + [S], np.ones(4)),
    }
    thresholds = [round(float(x), 3) for x in np.arange(.15, .551, .05)]
    output = {"common_sites": len(common), "gt_count": len(L), "variants": {}}
    for name, (frames, w) in variants.items():
        F = fuse(frames, w) if len(frames) > 1 else frames[0]
        settings = thresholds

        def make(t, fold):
            return fc(F[(F.score >= t) & (F.region.map(RF) == fold)])

        C = fold_counts(make, settings, folds, gt)
        honest, chosen, oracle, oracle_t, by_cls = nested(C)
        fixed = {str(t): float(__import__("koz3.nested", fromlist=["weighted"]).weighted(
            __import__("koz3.nested", fromlist=["_add_all"])._add_all(C[t].values()))) for t in thresholds}
        result = {
            "nested_score": float(honest), "selected_by_fold": {str(k): v for k, v in chosen.items()},
            "oracle_score": float(oracle), "oracle_threshold": oracle_t,
            "fixed_scores": fixed, "class_counts": {k: list(map(int, v)) for k, v in by_cls.items()},
            "candidate_count": len(F),
        }
        output["variants"][name] = result
        print(name, "nested", honest, "oracle", oracle, "chosen", chosen, flush=True)
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
