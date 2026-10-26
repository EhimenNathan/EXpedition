"""CPU-only optical morphology features for satellite kurgan OOF candidates.

Inputs are untouched held-out predictions and images. Labels are used only
after features are fixed, to train a fold-safe objectness filter. Nothing
here reads the competition's hidden test labels.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from shapely import STRtree
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sp_assemble_oof import models, region_folds
from sp_dual_oof import fuse_cls

NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
RADIUS = 48
yy, xx = np.mgrid[-RADIUS:RADIUS + 1, -RADIUS:RADIUS + 1]
rr = np.hypot(xx, yy)
rad_masks = [(rr < 6), ((rr >= 6) & (rr < 12)), ((rr >= 12) & (rr < 24)), ((rr >= 24) & (rr <= 48))]
angle = np.arctan2(yy, xx)
sectors = [[m & (angle >= -np.pi + j * np.pi / 4) & (angle < -np.pi + (j + 1) * np.pi / 4)
            for j in range(8)] for m in rad_masks[1:]]


def patch_features(arr, x, y):
    """Fixed physical-scale annuli plus directionality, in a 0.5m/px image."""
    _, H, W = arr.shape
    if not (0 <= x < W and 0 <= y < H):
        return None
    ys = np.clip(np.arange(y - RADIUS, y + RADIUS + 1), 0, H - 1)
    xs = np.clip(np.arange(x - RADIUS, x + RADIUS + 1), 0, W - 1)
    crop = np.asarray(arr[:, ys[:, None], xs[None, :]], dtype=np.float32)
    if crop.shape[0] < 3:
        return None
    rgb = crop[:3]
    valid = crop[3] > 0 if crop.shape[0] > 3 else np.ones(crop.shape[1:], bool)
    if not valid[RADIUS, RADIUS] or valid.mean() < 0.7:
        return None
    gray = .299 * rgb[0] + .587 * rgb[1] + .114 * rgb[2]
    feat = {"valid_frac_48": float(valid.mean()), "center_gray": float(gray[RADIUS, RADIUS])}
    means = []
    for i, mask in enumerate(rad_masks):
        v = gray[mask & valid]
        means.append(float(v.mean()) if len(v) else np.nan)
        feat[f"gray_mean_r{i}"] = means[-1]
        feat[f"gray_std_r{i}"] = float(v.std()) if len(v) else np.nan
        for c in range(3):
            q = rgb[c][mask & valid]
            feat[f"rgb{c}_mean_r{i}"] = float(q.mean()) if len(q) else np.nan
    for i in range(3):
        feat[f"radial_diff_{i}"] = means[i] - means[i + 1]
        feat[f"radial_absdiff_{i}"] = abs(means[i] - means[i + 1])
    for i, wedges in enumerate(sectors):
        v = np.array([gray[m & valid].mean() if np.any(m & valid) else np.nan for m in wedges])
        feat[f"angular_std_r{i + 1}"] = float(np.nanstd(v))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = rad_masks[2] & valid
    a, b, c = float(np.mean(gx[m] ** 2)), float(np.mean(gx[m] * gy[m])), float(np.mean(gy[m] ** 2))
    feat["gradient_energy"] = a + c
    feat["linear_anisotropy"] = float(np.hypot(a - c, 2 * b) / (a + c + 1e-4))
    # Radial derivatives are positive for outward-darkening objects and
    # negative for outward-brightening objects; fields tend to be directional.
    radial = (gx * xx + gy * yy) / (rr + 1)
    tangential = (-gx * yy + gy * xx) / (rr + 1)
    feat["radial_gradient_mean"] = float(radial[m].mean())
    feat["radial_gradient_abs"] = float(np.abs(radial[m]).mean())
    feat["tangential_gradient_abs"] = float(np.abs(tangential[m]).mean())
    return feat


def main():
    cache = Path(__file__).with_name("sp_morph_features.parquet")
    M = models(extra=list(NAMES[3:]))
    regions = sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
    F = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES])
    F = F[(F.cls.isin(K)) & (F.score >= .08)].reset_index(drop=True)
    print("candidates", len(F), flush=True)
    SC = pd.read_csv(ROOT / "kaggle_import/code/work/sp/scenes.csv")
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions) & L.class_name.isin(K)]
    RF = region_folds()
    rows = []
    for n, region in enumerate(regions):
        q = F[F.region == region].copy()
        if q.empty:
            continue
        q["fold"] = RF[region]
        q["y_obj"] = 0
        gt = list(L[L.region == region].geometry)
        tree = STRtree(gt) if gt else None
        if tree is not None:
            for ix, r in q.iterrows():
                p = Point(r.x, r.y)
                q.at[ix, "y_obj"] = int(any(gt[j].contains(p) for j in tree.query(p)))
        q["matched_scene"] = None
        q["scene_px_x"] = -1
        q["scene_px_y"] = -1
        for sc in SC[SC.site == region].itertuples():
            T = Affine(*json.loads(sc.T))
            tr = Transformer.from_crs(3857, sc.crs, always_xy=True)
            X, Y = tr.transform(q.x.to_numpy(), q.y.to_numpy())
            ix, iy = (~T) * (X, Y)
            ix = np.asarray(ix).round().astype(int)
            iy = np.asarray(iy).round().astype(int)
            ok = (ix >= RADIUS) & (iy >= RADIUS) & (ix < sc.W - RADIUS) & (iy < sc.H - RADIUS)
            # In overlapping scenes prefer the location farthest from an edge.
            margin = np.minimum.reduce([ix, iy, sc.W - 1 - ix, sc.H - 1 - iy])
            cur = q["scene_margin"].to_numpy() if "scene_margin" in q else np.full(len(q), -1)
            use = ok & (margin > cur)
            q.loc[use, "matched_scene"] = sc.scene
            q.loc[use, "scene_px_x"] = ix[use]
            q.loc[use, "scene_px_y"] = iy[use]
            q.loc[use, "scene_margin"] = margin[use]
        for scene, subset in q.groupby("matched_scene", dropna=True):
            arr = np.load(ROOT / "kaggle_import/code/work/sp/img" / f"{scene}.npy", mmap_mode="r")
            for r in subset.itertuples():
                z = patch_features(arr, int(r.scene_px_x), int(r.scene_px_y))
                if z is None:
                    continue
                d = {k: getattr(r, k) for k in ("region", "fold", "cls", "score", "x", "y", "s_kp", "s_kt", "y_obj")}
                d.update(z)
                rows.append(d)
        if (n + 1) % 20 == 0:
            print("regions", n + 1, "rows", len(rows), flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    print("saved", cache, len(out), "positive", int(out.y_obj.sum()), "folds", out.fold.value_counts().to_dict(), flush=True)

if __name__ == "__main__":
    main()
