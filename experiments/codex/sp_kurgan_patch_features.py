"""Extract CPU satellite patch descriptors for geographic OOF kurgan proposals.

Coordinates come from OOF predictions. The descriptors use only training
imagery pixels around each proposal, with no labels or site identifiers.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGES = ROOT / "kaggle_import/code/work/sp/img"
INPUT = HERE / "sp_agreement_oof.parquet"
OUTPUT = HERE / "sp_kurgan_patch_features.parquet"
RADII = (8, 16, 32, 64)


def names():
    return [f"r{r}_{x}" for r in RADII for x in
            ("red_mean", "green_mean", "blue_mean", "gray_std", "sat_mean", "edge_mean", "center_ring", "valid_frac")]


def descriptor(a, u, v):
    h, w = a.shape[1:]
    values = []
    for r in RADII:
        x0, x1 = max(0, u-r), min(w, u+r)
        y0, y1 = max(0, v-r), min(h, v+r)
        p = np.asarray(a[:, y0:y1, x0:x1])
        if not p.size:
            values.extend([np.nan] * 8)
            continue
        valid = p[3] > 0
        rgb = p[:3].transpose(1, 2, 0)
        if valid.sum() < 10:
            values.extend([np.nan] * 7 + [float(valid.mean())])
            continue
        bgr = rgb[:, :, ::-1].copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        edge = cv2.Laplacian(gray, cv2.CV_32F)
        yy, xx = np.indices(gray.shape)
        rr = np.hypot(xx-(u-x0), yy-(v-y0))
        center = (rr <= r/2) & valid
        ring = (rr >= r*.65) & (rr <= r*.95) & valid
        contrast = float(gray[center].mean() - gray[ring].mean()) if center.any() and ring.any() else np.nan
        values.extend([*(float(rgb[:,:,c][valid].mean()) / 255 for c in range(3)),
                       float(gray[valid].std()) / 255, float(sat[valid].mean()) / 255,
                       float(np.abs(edge[valid]).mean()) / 255, contrast / 255, float(valid.mean())])
    return values


def main():
    F = pd.read_parquet(INPUT)
    F = F[F.cls.isin(("kurgany_povrezhdennye", "kurgany_tselye")) & (F.score >= .08)].copy().reset_index(drop=True)
    S = pd.read_csv(ROOT / "kaggle_import/code/work/sp/scenes.csv")
    S = S.sort_values("valid_frac", ascending=False)
    features = np.full((len(F), len(names())), np.nan, np.float32)
    selected_scene = np.full(len(F), "", dtype=object)
    for ri, (region, idx) in enumerate(F.groupby("region").groups.items(), 1):
        idx = np.asarray(list(idx))
        remaining = np.ones(len(idx), bool)
        X, Y = F.loc[idx, ["x", "y"]].to_numpy().T
        for scene in S[S.site == region].itertuples():
            T = Affine(*json.loads(scene.T))
            transformer = Transformer.from_crs(3857, scene.crs, always_xy=True)
            sx, sy = transformer.transform(X, Y)
            u, v = (~T) * (sx, sy)
            u, v = np.asarray(u).astype(int), np.asarray(v).astype(int)
            usable = remaining & (u >= 0) & (v >= 0) & (u < scene.W) & (v < scene.H)
            if not usable.any():
                continue
            a = np.load(IMAGES / (scene.scene + ".npy"), mmap_mode="r")
            for j in np.where(usable)[0]:
                features[idx[j]] = descriptor(a, u[j], v[j])
                selected_scene[idx[j]] = scene.scene
                remaining[j] = False
        if ri % 20 == 0:
            print(f"regions {ri}/{F.region.nunique()}, patches {np.isfinite(features[:, 0]).sum()}/{len(F)}", flush=True)
    out = pd.DataFrame(features, columns=names())
    out.insert(0, "region", F.region)
    out.insert(1, "cls", F.cls)
    out.insert(2, "score", F.score)
    out["scene_found"] = selected_scene != ""
    out.to_parquet(OUTPUT, index=False)
    print(f"wrote {OUTPUT}, {len(out)} rows, scene found {out.scene_found.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
