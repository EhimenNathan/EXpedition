"""Inference-only probe for the FCOS per-tile output cap on held-out Ae folds.

Uses the already trained geographic OOF checkpoints.  It changes only
``detections_per_img`` before the accurate sliding-window/TTA inference path.
"""
import argparse
import os
import sys

import pandas as pd

ROOT = "/home/nathan/Expedition Competition"
sys.path.insert(0, ROOT)

from koz3.ae_det_train import CLS, chips, detect_image, image, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--region")
    ap.add_argument("--tag", default="ae_fcos")
    a = ap.parse_args()

    from pyproj import Transformer
    from rasterio.transform import Affine
    from koz3.sp_fuse import fuse

    model = load_model(os.path.join(ROOT, f"work/ae/{a.tag}_f{a.fold}.pt"))
    model.detections_per_img = a.cap
    model.score_thresh = a.score_thresh
    model.topk_candidates = a.topk
    rows = []
    held = [c for c in chips() if c["fold"] == a.fold and (a.region is None or c["region"] == a.region)]
    for n, c in enumerate(held, 1):
        transform = Affine(*c["transform"])
        project = Transformer.from_crs(c["crs"], 3857, always_xy=True)
        det = detect_image(model, image(c), fast=False)
        for cl, cx, cy, _, _, score in det:
            x, y = project.transform(*(transform * (cx, cy)))
            rows.append((c["region"], CLS[cl], score, x, y, a.fold))
        print(n, len(held), c["region"], len(det), len(rows), flush=True)
    raw = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y", "fold"])
    merged = fuse([raw[["region", "cls", "score", "x", "y"]]], [1.0], match_m=4.0)
    merged["fold"] = a.fold
    suffix = ("_" + a.region.split('_',1)[0]) if a.region else ""
    out = f"/home/nathan/Expedition Competition/Codex Work/ae_cap_f{a.fold}_c{a.cap}_s{a.score_thresh:g}{suffix}.parquet"
    merged.to_parquet(out)
    print("saved", out, len(raw), "raw", len(merged), "merged", flush=True)


if __name__ == "__main__":
    main()
