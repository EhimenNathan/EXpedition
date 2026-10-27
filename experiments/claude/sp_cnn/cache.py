"""Cache every satellite scene at 0.5 m GSD (uint8, as sp_dense/extract.read_gsd) + its labelled objects in pixel
coordinates, for CPU CNN training (train.py).

  python experiments/claude/sp_cnn/cache.py
work/sp_cnn/img/<idx>.npy  (H, W, 3) uint8 (padded to a multiple of 16)
work/sp_cnn/meta/<idx>.npz T (affine of the image), crs, valid (h, w), region, tif,
                           objs (n, 5): row, col, sigma_px, class id (0 kurgan, 1 architecture, 2 fortification), damaged
"""
import glob
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location("sp_dense_extract", os.path.join(HERE, "..", "sp_dense", "extract.py"))
SP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SP)
OUT = os.path.join(ROOT, "work/sp_cnn")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
CID = {K[0]: 0, K[1]: 0, "arkhitektury": 1, "fortifikatsii": 2}


def main():
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True); os.makedirs(os.path.join(OUT, "meta"), exist_ok=True)
    for k, path in enumerate(SP.scenes()):
        f = os.path.join(OUT, "meta", f"{k:03d}.npz")
        if os.path.exists(f):
            continue
        img, T, crs, (h, w) = SP.read_gsd(path)
        region = os.path.basename(os.path.dirname(os.path.dirname(path)))
        g = L[(L.region == region) & L.class_name.isin(list(CID))].to_crs(crs)
        objs = []
        inv = ~T
        for geom, c in zip(g.geometry, g.class_name):
            col, row = inv * geom.representative_point().coords[0]
            if -32 <= row < h + 32 and -32 <= col < w + 32:
                objs.append((row, col, max(4.0, 0.2 * np.sqrt(geom.area)) / SP.GSD, CID[c], float(c == K[0])))
        np.save(os.path.join(OUT, "img", f"{k:03d}.npy"), img)
        np.savez(f, T=np.array(T[:6]), crs=crs.to_wkt(), valid=np.array([h, w]), region=region, tif=os.path.basename(path),
                 objs=np.array(objs, np.float32).reshape(-1, 5))
        print(f"{k + 1} {region[:24]} {img.shape} objs {len(objs)}", flush=True)


if __name__ == "__main__":
    main()
