"""Cache every LiDAR region as a 0.5 m relief image (li_dense/extract.relief_image: hillshade, local relief 5 / 15 m)
plus its labelled objects in pixel coordinates, in the sp_cnn format, so train.py runs unchanged:
  CNN_WORK=work/li_cnn KOZ3_MOD=Li python experiments/claude/sp_cnn/train.py ...
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location("li_dense_extract", os.path.join(HERE, "..", "li_dense", "extract.py"))
LI = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(LI)
OUT = os.path.join(ROOT, "work/li_cnn")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
CID = {K[0]: 0, K[1]: 0, "arkhitektury": 1, "fortifikatsii": 2}


def main():
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "Li"]
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True); os.makedirs(os.path.join(OUT, "meta"), exist_ok=True)
    for k, path in enumerate(LI.regions()):
        f = os.path.join(OUT, "meta", f"{k:03d}.npz")
        if os.path.exists(f):
            continue
        img, T, crs, (h, w) = LI.read_region(path)
        region = os.path.basename(path)[:-4]
        g = L[(L.region == region) & L.class_name.isin(list(CID))].to_crs(crs)
        inv = ~T; objs = []
        for geom, c in zip(g.geometry, g.class_name):
            col, row = inv * geom.representative_point().coords[0]
            if -32 <= row < h + 32 and -32 <= col < w + 32:
                objs.append((row, col, max(4.0, 0.2 * np.sqrt(geom.area)) / 0.5, CID[c], float(c == K[0])))
        np.save(os.path.join(OUT, "img", f"{k:03d}.npy"), img)
        np.savez(f, T=np.array(T[:6]), crs=crs, valid=np.array([h, w]), region=region, tif=region,
                 objs=np.array(objs, np.float32).reshape(-1, 5))
        print(f"{k + 1} {region[:24]} {img.shape} objs {len(objs)}", flush=True)


if __name__ == "__main__":
    main()
