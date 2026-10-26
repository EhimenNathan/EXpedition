"""Independent satellite threshold audit on geographic out-of-fold predictions.

All source files are read-only. For each held-out geographic fold, select thresholds
using counts from the other folds only, then pool held-out counts with the official
metric. This is an audit, not a deployed inference modification.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from koz3.labels import to_gt_geojson
from koz3.nested import weighted
from koz3.score import WEIGHTS, score
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc


def add_counts(rows):
    out = np.zeros(3, dtype=int)
    for row in rows:
        out += row
    return out


def main():
    sources = models()
    regions = sorted(set.intersection(*(set(x.region) for x in sources.values())))
    sources = {k: v[v.region.isin(regions)] for k, v in sources.items()}
    fused = fuse(list(sources.values()), np.ones(len(sources)))
    folds_map = region_folds()
    folds = sorted({folds_map[r] for r in regions})
    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    labels = labels[(labels["mod"] == "SpOr") & labels.region.isin(regions)]
    thresholds = [round(float(x), 3) for x in np.arange(0.10, 0.601, 0.025)]
    classes = list(WEIGHTS)
    counts = {}
    for cls in classes:
        C = {}
        for fold in folds:
            regs = {r for r in regions if folds_map[r] == fold}
            gt = to_gt_geojson(labels[(labels.region.isin(regs)) & (labels.class_name == cls)])
            subset = fused[(fused.region.isin(regs)) & (fused.cls == cls)]
            C[fold] = {}
            for threshold in thresholds:
                pred = fc(subset[subset.score >= threshold])
                v = score(pred, gt, [cls])[1][cls]
                C[fold][threshold] = [int(v[k]) for k in ("tp", "fp", "fn")]
        counts[cls] = C
        print(cls, "done", flush=True)

    results = {}
    for cls, by_fold in counts.items():
        chosen = {}
        held = []
        for fold in folds:
            best = max(thresholds, key=lambda t: weighted({cls: add_counts(by_fold[f][t] for f in folds if f != fold)}))
            chosen[str(fold)] = best
            held.append(by_fold[fold][best])
        nested = add_counts(held)
        base = add_counts(by_fold[f][0.25] for f in folds)
        oracle_t = max(thresholds, key=lambda t: weighted({cls: add_counts(by_fold[f][t] for f in folds)}))
        results[cls] = {
            "baseline": base.tolist(), "nested": nested.tolist(),
            "baseline_f1": weighted({cls: base}), "nested_f1": weighted({cls: nested}),
            "oracle_threshold": oracle_t, "selected_by_fold": chosen,
            "label_count_by_fold": {str(f): int(by_fold[f][0.25][0] + by_fold[f][0.25][2]) for f in folds},
        }
    base = {c: v["baseline"] for c, v in results.items()}
    nested = {c: v["nested"] for c, v in results.items()}
    output = {"baseline_score": weighted(base), "nested_score": weighted(nested), "classes": results}
    path = Path(__file__).with_name("sp_threshold_audit.json")
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
