"""Satellite D-FINE (previous team, trained on SpOr only) applied to Ae chips; for each Ae site the fold model that
held out that site's SpOr labels is used (no location leakage). Scored nested against Ae labels."""
import os, sys, json
import numpy as np, pandas as pd, cv2, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    from pyproj import Transformer
    from rasterio.transform import Affine
    from koz3.det_train import load_eager, detect_chip
    from koz3.sp_train import load_chips
    from sp_assemble_oof import region_folds
    PREV = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
    RF = region_folds()
    chips = load_chips("Ae")
    models = {}
    rows = []
    for c in chips:
        f = RF.get(c["region"], 0)
        if f not in models:
            models[f] = load_eager("ustc-community/dfine-large-coco", os.path.join(ROOT, f"kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/fold{f}.pt"))
        rgb = cv2.imread(os.path.join(ROOT, "data/ae_prep/chips", c["file"]))[:, :, ::-1]
        t = Affine(*c["transform"]); tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
        for cl, cx, cy, bw, bh, s in detect_chip(models[f], rgb, tile=640, ov=160):
            X, Y = tr.transform(*(t * (cx, cy)))
            rows.append((c["region"], PREV[cl], s, X, Y))
    df = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])
    df.to_parquet(os.path.join(ROOT, "work/ae/oof_spdet_cands.parquet")); print(len(df), "candidates")


if __name__ == "__main__":
    main()
