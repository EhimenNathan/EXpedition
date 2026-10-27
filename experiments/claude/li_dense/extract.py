"""Dense DINOv3-SAT features for every LiDAR region from scratch (CPU): 0.5 m DTM (koz3.dtm via li_stream_dtm.py) →
3-channel relief image → the satellite tiling / backbone of sp_dense/extract.py → token grid (8 m) per region.

Relief image (uint8): R = multi-directional hillshade (azimuths 315/45/135/225°, altitude 45° — the organisers drew
the LiDAR labels on hillshades), G = local relief 5 m (DTM − Gaussian σ 5 m, ±1 m), B = local relief 15 m (±2 m).
Invalid cells (no ground within 3 m) are set to mid-grey and marked invalid.

  python experiments/claude/li_dense/extract.py pca | run
Output format identical to sp_dense (work/li_dense/scenes/<idx>.npz), so heads.py / evaluate.py run unchanged with
SP_DENSE_OUT=work/li_dense KOZ3_MOD=Li.
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("sp_dense_extract", os.path.join(HERE, "..", "sp_dense", "extract.py"))
SP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SP)

OUT = os.path.join(ROOT, "work/li_dense")
DTM = os.path.join(ROOT, "work/li_dome/dtm")


def relief_image(d, v, res):
    from scipy.ndimage import gaussian_filter
    d = d.astype(np.float32)
    dz = gaussian_filter(d, 0.75)
    gy, gx = np.gradient(dz, res)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    alt = np.radians(45); hs = 0
    for az in (315, 45, 135, 225):
        a = np.radians(az)
        hs = hs + np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(a - aspect), 0, 1)
    hs = hs / 4
    l5 = np.clip((d - gaussian_filter(d, 5 / res)) / 1.0, -1, 1)
    l15 = np.clip((d - gaussian_filter(d, 15 / res)) / 2.0, -1, 1)
    hs = np.clip((hs - 0.45) / 0.5, 0, 1)                     # flat ground ≈ 0.71 → mid-range, shadows / lit slopes spread
    img = np.stack([hs * 255, (l5 + 1) * 127.5, (l15 + 1) * 127.5], -1)
    img[~v] = 128
    return np.clip(img, 0, 255).astype(np.uint8)


def regions():
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    li = set(L[L["mod"] == "Li"].region)
    return [p for p in sorted(glob.glob(os.path.join(DTM, "*.npz"))) if os.path.basename(p)[:-4] in li]


def read_region(path):
    """→ padded uint8 image, rasterio Affine of the image, CRS string, valid (h, w) in pixels, valid mask."""
    from rasterio.transform import Affine
    z = np.load(path); d, v = z["d"], z["v"].astype(bool); x0, ytop, res = z["origin"]
    if abs(res - SP.GSD) > 1e-6:
        raise ValueError(f"{path}: DTM res {res} ≠ {SP.GSD}")
    reg = os.path.basename(path)[:-4]
    crs = json.load(open(os.path.join(ROOT, "data/train-expds-3", reg, "UTM.json")))["crs"].split("::")[-1]
    img = relief_image(d, v, res); h, w = img.shape[:2]
    H = max(SP.TILE * SP.PATCH, int(np.ceil(h / SP.PATCH)) * SP.PATCH); W = max(SP.TILE * SP.PATCH, int(np.ceil(w / SP.PATCH)) * SP.PATCH)
    pad = np.full((H, W, 3), 128, np.uint8); pad[:h, :w] = img
    return pad, Affine(res, 0, x0, 0, -res, ytop), f"EPSG:{crs}", (h, w)


def fit_pca(n_tiles=400, seed=0):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed); bb = SP.Backbone(); S = regions(); X = []
    per = max(1, n_tiles // len(S)) + 1
    for k, path in enumerate(S):
        img = read_region(path)[0]
        gh, gw = img.shape[0] // SP.PATCH, img.shape[1] // SP.PATCH
        jobs = [(r, c) for r in SP.starts(gh) for c in SP.starts(gw)]
        for j in rng.choice(len(jobs), min(per, len(jobs)), replace=False):
            r, c = jobs[j]
            X.append(bb([img[r * SP.PATCH:(r + SP.TILE) * SP.PATCH, c * SP.PATCH:(c + SP.TILE) * SP.PATCH]])[0].reshape(-1, 1024)[::4])
        print(f"pca sample {k + 1}/{len(S)}", flush=True)
    p = PCA(SP.PCA_DIM, random_state=0).fit(np.concatenate(X))
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "pca.npz"), mean=p.mean_.astype(np.float32), comp=p.components_.astype(np.float32), var=p.explained_variance_ratio_)
    print(f"PCA {SP.PCA_DIM}: explained variance {p.explained_variance_ratio_.sum():.3f}")


def run():
    P = np.load(os.path.join(OUT, "pca.npz")); mean, comp = P["mean"], P["comp"]
    project = lambda t: (t - mean) @ comp.T
    bb = SP.Backbone(); S = regions(); os.makedirs(os.path.join(OUT, "scenes"), exist_ok=True); t0 = time.time()
    for k, path in enumerate(S):
        f = os.path.join(OUT, "scenes", f"{k:03d}.npz")
        if os.path.exists(f):
            continue
        img, T, crs, (h, w) = read_region(path)
        tok = SP.scene_tokens(bb, img, project)
        Tt = T * T.scale(SP.PATCH, SP.PATCH)
        reg = os.path.basename(path)[:-4]
        np.savez(f + ".tmp.npz", f=tok, T=np.array(Tt[:6]), crs=crs, valid=np.array([h, w]) / SP.PATCH, region=reg, tif=reg)
        os.replace(f + ".tmp.npz", f)
        print(f"{k + 1}/{len(S)} {reg[:30]} grid {tok.shape[:2]} {time.time() - t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    {"pca": fit_pca, "run": run}[sys.argv[1]]()
