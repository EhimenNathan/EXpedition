#!/usr/bin/env python3
"""Prospective fold diagnostic for the satellite P2 member.

This is a go/no-go screen only.  Any positive fold result must be repeated on
the other three geographic folds and selected without each held-out fold.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    path = Path(__file__).resolve().parent / f"fold{args.fold}/oof_candidates.parquet"
    yolo = pd.read_parquet(path)[["region", "cls", "score", "x", "y"]]
    names = ["dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3"]
    members = models(extra=names[3:])
    rf = region_folds()
    sites = {r for r, f in rf.items() if f == args.fold}
    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    labels = labels[(labels["mod"] == "SpOr") & labels.region.isin(sites)]
    gt = to_gt_geojson(labels)
    base_members = [members[n][members[n].region.isin(sites)] for n in names]
    yolo = yolo[yolo.region.isin(sites)]
    versions = {"p2": yolo, "base6": fuse(base_members, np.ones(6))}
    for weight in (0.25, 0.5, 1.0):
        versions[f"base6+p2@{weight}"] = fuse(base_members + [yolo], np.array([1.] * 6 + [weight]))
    print("fold", args.fold, "sites", len(sites), "GT", len(labels), "P2 candidates", len(yolo))
    for name, pred in versions.items():
        best = None
        for threshold in (.10, .15, .20, .25, .30, .35, .40, .50):
            q = pred[pred.score >= threshold]
            value, details = score(fc(q), gt)
            item = (value, threshold, len(q), details)
            best = item if best is None or item[0] > best[0] else best
        print(f"{name:16s} best={best[0]:.5f} threshold={best[1]:.2f} n={best[2]}")


if __name__ == "__main__":
    main()
