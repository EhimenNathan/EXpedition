"""Export layer-aware Ae FCOS OOF candidates with physical box-shape metadata.

The deployed pipeline emits centre points, but FCOS predicts a box first.  Keeping
the box width, height, aspect ratio and chip-edge clearance lets us test physical
filters on geographically held-out folds before changing deployment.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from koz3.ae_det_train import chips, detect_image, image, load_model
from koz3.sp_fuse import fuse
from koz3.sp_train import CLS


def main():
    from pyproj import Transformer
    from rasterio.transform import Affine
    from scipy.spatial import cKDTree

    C = chips()
    rows = []
    for fold in range(4):
        m = load_model(os.path.join(ROOT, f"work/ae/ae_fcos_f{fold}.pt"))
        for c in (q for q in C if q["fold"] == fold):
            t = Affine(*c["transform"])
            px = float(np.hypot(t.a, t.d)); py = float(np.hypot(t.b, t.e))
            tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
            for cl, cx, cy, bw, bh, score in detect_image(m, image(c), fast=True):
                X, Y = tr.transform(*(t * (cx, cy)))
                edge = min(cx - bw / 2, cy - bh / 2, c["w"] - cx - bw / 2, c["h"] - cy - bh / 2)
                rows.append((c["region"], c["scene"], CLS[cl], score, X, Y, fold,
                             bw * px, bh * py, edge * min(px, py)))
        print("raw fold", fold, len(rows), flush=True)

    D = pd.DataFrame(rows, columns=["region", "scene", "cls", "score", "x", "y", "fold",
                                        "bw_m", "bh_m", "edge_m"])
    D["unit"] = D.region + "\x1f" + D.scene
    Z = D.rename(columns={"region": "base_region", "unit": "region"})
    F = fuse([Z[["region", "cls", "score", "x", "y"]]], np.ones(1), match_m=4.0)

    # Copy box metadata from the nearest raw member of each fused object.  The
    # highest-score member normally defines the fused score; nearest is stable
    # even when score ties are rounded to float16.
    meta = []
    for (unit, cls), g in Z.groupby(["region", "cls"], sort=False):
        k = F[(F.region == unit) & (F.cls == cls)]
        if len(k) == 0:
            continue
        tree = cKDTree(g[["x", "y"]].to_numpy())
        _, ix = tree.query(k[["x", "y"]].to_numpy(), k=1)
        q = g.iloc[np.asarray(ix)][["bw_m", "bh_m", "edge_m", "fold"]].reset_index(drop=True)
        q.insert(0, "_fidx", k.index.to_numpy())
        meta.append(q)
    M = pd.concat(meta, ignore_index=True).set_index("_fidx")
    F = F.join(M)
    parts = F.region.str.split("\x1f", n=1, expand=True)
    F["region"], F["scene"] = parts[0], parts[1]
    F["aspect"] = np.maximum(F.bw_m, F.bh_m) / np.maximum(np.minimum(F.bw_m, F.bh_m), 1e-3)
    F["box_area_m2"] = F.bw_m * F.bh_m
    out = os.path.join(ROOT, "work/ae/oof_ae_fcos_fast_layers_boxmeta.parquet")
    F.to_parquet(out)
    print(len(D), "raw ->", len(F), "fused", out, flush=True)


if __name__ == "__main__":
    main()
