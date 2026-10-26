"""Stack our geophysics CNN (v2, detrended input) into the previous team's finds pipeline (physics candidates +
LightGBM, leave-one-region-out).

Adds to every cached candidate row: cnn = OOF finds-heatmap value at the candidate, cnn_max1m = max within 1 m.
The OOF maps come from models that never saw the candidate's region, so the feature is out-of-fold; their model
stage then runs unchanged (LORO, thresholds chosen on the other regions) on kaggle_import/code/work/geo/finds_v3s.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from scipy import ndimage as ndi

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "kaggle_import/code/work/geo/finds_v3")
DST = os.path.join(ROOT, "kaggle_import/code/work/geo/finds_v3s")


def main(tag="_v2"):
    from koz3.geo_train import PREP
    P = pickle.load(open(os.path.join(ROOT, f"work/geo/oof_probs{tag}.pkl"), "rb"))
    maps = {}
    for name, (prob, _) in P.items():
        meta = json.loads(str(np.load(os.path.join(PREP, name + ".npz"))["meta"]))
        if meta["method"] != "MagGr":
            continue
        key = "" if meta["key"] == "серый" else meta["key"]
        hm = prob[0].astype(np.float32)
        maps[f"{meta['region']}|{key}"] = (hm, ndi.maximum_filter(hm, 21), Affine(*meta["transform"]), meta["crs"])
    os.makedirs(os.path.join(DST, "units"), exist_ok=True)
    for p in sorted(glob.glob(os.path.join(SRC, "units", "*.parquet"))):
        df = pd.read_parquet(p)
        uid = df.uid.iloc[0] if len(df) else None
        if uid not in maps:
            print("no CNN map for", uid); continue
        hm, hmx, t, crs = maps[uid]
        inv = ~t
        col, row = inv * (df.x.values, df.y.values)
        r = np.clip(np.round(row).astype(int), 0, hm.shape[0] - 1); c = np.clip(np.round(col).astype(int), 0, hm.shape[1] - 1)
        df["cnn"] = hm[r, c]; df["cnn_max1m"] = hmx[r, c]
        df.to_parquet(os.path.join(DST, "units", os.path.basename(p)))
    import shutil
    shutil.copy(os.path.join(SRC, "gt.json"), os.path.join(DST, "gt.json"))
    print("stacked units →", DST)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
