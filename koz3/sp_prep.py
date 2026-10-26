"""Optical detector training chips (SpOr by default; --mod Ae / Or) with boxes, grouped folds.

data/<mod>_prep/chips/*.jpg  +  data/<mod>_prep/ann.json  (list of chips: file, region, scene, fold, kind,
transform (chip px → scene CRS), crs, boxes [x0,y0,x1,y1] px, classes, valid_frac).
Folds: regions clustered by 20 km single linkage (neighbouring sites never straddle train/val), 4 folds.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3.labels import DATA, region_utm  # noqa: E402
from koz3 import sp_io  # noqa: E402

CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
MOD_GLOB = {"SpOr": "04_*/*.tif", "Ae": "03_*/*.tif", "Or": "05_*/*.tif"}


def site_folds(L, n_folds=4, km=20.0, seed=0):
    """region → fold via 20 km single-linkage clusters of label centroids, greedily balanced by label count."""
    from scipy.cluster.hierarchy import fcluster, linkage
    c = L.to_crs(3857).geometry.centroid
    g = L.assign(x=c.x, y=c.y).groupby("region").agg(x=("x", "mean"), y=("y", "mean"), n=("x", "size"))
    lat = np.degrees(2 * np.arctan(np.exp(g.y / 6378137.0)) - np.pi / 2)
    xy = np.c_[g.x * np.cos(np.radians(lat)), g.y * np.cos(np.radians(lat))] / 1000.0
    cl = fcluster(linkage(xy, "single"), km, "distance") if len(g) > 1 else np.ones(1, int)
    g["cl"] = cl
    sizes = g.groupby("cl").n.sum().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    load = np.zeros(n_folds); fold_of_cl = {}
    for cid, n in sizes.items():
        k = int(np.argmin(load + rng.uniform(0, 1e-3, n_folds)))
        fold_of_cl[cid] = k; load[k] += n
    return {r: fold_of_cl[cl] for r, cl in g.cl.items()}


def boxes_in(lab, transform, crs, W, H, min_inside=0.5):
    inv = ~transform
    bx, cls = [], []
    for g, c in zip(lab.to_crs(crs).geometry, lab.class_name):
        if g is None or g.is_empty:
            continue
        x0, y0, x1, y1 = g.bounds
        pts = np.array([inv * (x0, y0), inv * (x1, y1)])
        px0, px1 = sorted(pts[:, 0]); py0, py1 = sorted(pts[:, 1])
        full = box(px0, py0, px1, py1)
        clip = full.intersection(box(0, 0, W, H))
        if clip.is_empty or clip.area < min_inside * max(full.area, 1e-6):
            continue
        cx0, cy0, cx1, cy1 = clip.bounds
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        bx.append([float(cx0), float(cy0), float(cx1), float(cy1)]); cls.append(CLASSES.index(c))
    return bx, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", default="SpOr")
    ap.add_argument("--res", type=float, default=sp_io.TILE_RES)
    ap.add_argument("--near_m", type=float, default=0, help="keep only tiles within this distance of a label (partly annotated sheets)")
    ap.add_argument("--append", action="store_true", help="merge into an existing ann.json (streamed preps)")
    a = ap.parse_args()
    out = os.path.join(DATA, "..", f"{a.mod.lower()}_prep"); os.makedirs(os.path.join(out, "chips"), exist_ok=True)
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    L = L[(L["mod"] == a.mod) & L.class_name.isin(CLASSES)].explode(index_parts=False)
    folds = site_folds(L)
    ann = []
    for f in sorted(glob.glob(os.path.join(DATA, "*", MOD_GLOB[a.mod]))):
        region = os.path.relpath(f, DATA).split(os.sep)[0]
        lab = L[L.region == region]
        if region not in folds:
            print("no labels, skip", f); continue
        try:
            src = rasterio.open(f)
        except Exception as e:
            print("unreadable", f, e); continue
        with src:
            crs = src.crs if (src.crs is not None and src.crs.to_epsg()) else rasterio.crs.CRS.from_user_input(region_utm(os.path.join(DATA, region)))
            sb = box(*rasterio.transform.array_bounds(src.height, src.width, src.transform)[::1])
            stem = os.path.basename(f)[:-4]
            if sp_io.is_small(src):
                rgb, valid, t = sp_io.read_small(src)
                chips = [(rgb, valid, t, "small")]
            else:
                W, H, tg = sp_io.big_grid(src, a.res)
                chips = []
                near = None
                if a.near_m > 0 and len(lab):
                    near = lab.to_crs(crs).buffer(a.near_m).union_all()
                for x, y, w, h in sp_io.tiles(W, H):
                    if near is not None:
                        tb = rasterio.transform.array_bounds(h, w, tg * rasterio.Affine.translation(x, y))
                        if not box(tb[0], tb[1], tb[2], tb[3]).intersects(near):
                            continue
                    try:
                        rgb, valid = sp_io.read_window_resampled(src, x, y, w, h, W, H)
                    except rasterio.errors.RasterioIOError as e:   # corrupt strips (e.g. 093 Or): skip the tile
                        print("  unreadable tile", x, y, str(e)[:80], flush=True)
                        continue
                    if valid.mean() < 0.1:
                        continue
                    chips.append((rgb, valid, tg * rasterio.Affine.translation(x, y), "tile"))
            for i, (rgb, valid, t, kind) in enumerate(chips):
                h, w = rgb.shape[:2]
                bx, cl = boxes_in(lab, t, crs, w, h)
                name = f"{region}__{stem}__{i:04d}.jpg"
                cv2.imwrite(os.path.join(out, "chips", name), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                ann.append(dict(file=name, region=region, scene=stem, fold=int(folds[region]), kind=kind,
                                transform=list(t)[:6], crs=crs.to_string(), boxes=bx, classes=cl,
                                valid_frac=float(valid.mean()), w=w, h=h))
            print(f"{region:36s} {stem[:40]:40s} {len(chips):4d} chips {sum(len(c['boxes']) for c in ann if c['scene'] == stem):5d} boxes", flush=True)
    af = os.path.join(out, "ann.json")
    if a.append and os.path.exists(af):
        done = {c["region"] for c in ann}
        ann = [c for c in json.load(open(af)) if c["region"] not in done] + ann
    json.dump(ann, open(af, "w"), ensure_ascii=False)
    print(len(ann), "chips;", sum(len(c["boxes"]) for c in ann), "boxes")


if __name__ == "__main__":
    main()
