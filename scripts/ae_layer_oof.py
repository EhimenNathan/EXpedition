"""Scene-preserving Ae OOF audit for the organiser's per-image reference rule.

Each source image is an independent layer.  Ground-truth polygons whose
representative point falls in two image footprints are duplicated, and detector
candidates are kept separate by source image.  Thresholds are selected on the
other geographic folds, class by class, including the option to emit nothing.
"""
from __future__ import annotations

import glob
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

ROOT = "/home/nathan/Expedition Competition"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from koz3.labels import to_gt_geojson
from koz3.score import WEIGHTS, score
from sp_fusion_oof import fc

THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.10)


def layer_gt(labels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    parts = []
    for region, group in labels.groupby("region"):
        rasters = sorted(glob.glob(os.path.join(ROOT, "data/train-expds-3", region, "03_*", "*.tif")))
        for path in rasters:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                local = group.to_crs(src.crs)
                footprint = box(*src.bounds)
                keep = local.geometry.representative_point().map(footprint.covers).to_numpy()
            if keep.any():
                selected = group.loc[keep].copy()
                selected["scene"] = os.path.splitext(os.path.basename(path))[0]
                parts.append(selected)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=labels.crs)


def f1(counts):
    tp, fp, fn = counts
    return 2 * tp / max(2 * tp + fn + 0.5 * fp, 1e-12)


def main(path: str):
    cand = pd.read_parquet(path)
    labels = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    labels = labels[(labels["mod"] == "Ae") & labels.region.isin(cand.region.unique())]
    layered = layer_gt(labels)
    fold_of = cand.groupby("region").fold.first().astype(int).to_dict()
    folds = sorted(set(fold_of.values()))
    gt = {
        fold: to_gt_geojson(layered[layered.region.map(fold_of) == fold])
        for fold in folds
    }
    print(f"base GT {len(labels)}, layer GT {len(layered)}, candidates {len(cand)}, folds {folds}")

    classes = sorted(labels.class_name.unique())
    counts = {}
    for cls in classes:
        dcls = cand[cand.cls == cls]
        for threshold in THRESHOLDS:
            for fold in folds:
                pred = dcls[(dcls.score >= threshold) & (dcls.region.map(fold_of) == fold)]
                _, result = score(fc(pred), gt[fold], classes=[cls])
                v = result[cls]
                counts[(cls, threshold, fold)] = np.array((v["tp"], v["fp"], v["fn"]), dtype=int)

    total = {}; choices = {}
    for cls in classes:
        total[cls] = np.zeros(3, dtype=int); choices[cls] = []
        for fold in folds:
            train_counts = {
                threshold: sum((counts[(cls, threshold, other)] for other in folds if other != fold), np.zeros(3, dtype=int))
                for threshold in THRESHOLDS
            }
            threshold = max(THRESHOLDS, key=lambda t: f1(train_counts[t]))
            choices[cls].append(threshold)
            total[cls] += counts[(cls, threshold, fold)]

    present = [cls for cls in classes if total[cls][0] + total[cls][2] > 0]
    result = sum(WEIGHTS[cls] * f1(total[cls]) for cls in present) / sum(WEIGHTS[cls] for cls in present)
    print(f"nested layer-aware score {result:.6f}")
    for cls in present:
        print(f"{cls:24s} F1 {f1(total[cls]):.6f} tp/fp/fn {tuple(total[cls])} thresholds {choices[cls]}")

    # Oracle is diagnostic only.
    for cls in present:
        pooled = {
            threshold: sum((counts[(cls, threshold, fold)] for fold in folds), np.zeros(3, dtype=int))
            for threshold in THRESHOLDS
        }
        threshold = max(THRESHOLDS, key=lambda t: f1(pooled[t]))
        print(f"oracle {cls:17s} {threshold:.2f} F1 {f1(pooled[threshold]):.6f} {tuple(pooled[threshold])}")


if __name__ == "__main__":
    main(sys.argv[1])
