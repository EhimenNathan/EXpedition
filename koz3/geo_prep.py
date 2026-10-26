"""Geophysics prep: every MagGr / EtGr / TerFl raster resampled to RES m/px in its UTM CRS, with targets.

Per raster → data/geo_prep/<region>__<stem>.npz:
  img (3,H,W) uint8 · obj (H,W) uint8 object_poly mask · valid (H,W) uint8 (inside survey bounds, outside noise)
  finds (N,2) float pixel coords (MagGr only; NaN-free) · has_finds flag · meta (transform, crs, key, method)
Raster ↔ labels: key = name part after _<Method>_ (e.g. 'серый', 'цвет_2', 'карта_04_0.75', 'профиль_16').
MagGr finds/objects with an empty key belong to every MagGr map of the region except keyed ones (цвет_2).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3.labels import DATA, region_utm  # noqa: E402

RES = 0.1
OUT = os.path.join(DATA, "..", "geo_prep")
METHODS = ("MagGr", "EtGr", "TerFl")


def raster_key(path):
    stem = os.path.basename(path)[:-4]
    m = re.search(r"_(MagGr|EtGr|TerFl)(?:_(.*))?$", stem)
    return m.group(1), (m.group(2) or "")


def read_resampled(path, res=RES, crs_fallback=None):
    """Raster → (img uint8 (3,H,W), alpha-valid (H,W), transform, crs) at `res` m/px in the raster CRS."""
    with rasterio.open(path) as s:
        crs = s.crs if (s.crs is not None and s.crs.to_epsg()) else rasterio.crs.CRS.from_user_input(crs_fallback)
        b = s.bounds
        W = max(8, int(round((b.right - b.left) / res))); H = max(8, int(round((b.top - b.bottom) / res)))
        dst_t = Affine(res, 0, b.left, 0, -res, b.top)
        n = s.count
        arr = np.zeros((n, H, W), np.float32)
        for i in range(n):
            reproject(rasterio.band(s, i + 1), arr[i], src_transform=s.transform, src_crs=crs, dst_transform=dst_t,
                      dst_crs=crs, resampling=Resampling.average if s.res[0] < res else Resampling.bilinear)
    if n >= 4:
        valid = arr[3] > 127
    else:
        valid = arr[:3].sum(0) > 0
    img = arr[:3] if n >= 3 else np.repeat(arr[:1], 3, 0)
    if img.max() > 255:
        lo, hi = np.percentile(img[:, valid], [1, 99]) if valid.any() else (img.min(), img.max())
        img = (img - lo) / max(hi - lo, 1e-6) * 255
    return np.clip(img, 0, 255).astype(np.uint8), valid, dst_t, crs


def read_geojson_polys(path, crs):
    """Polygons of a GeoJSON file reprojected to `crs` (json + shapely + pyproj only: no geopandas at inference)."""
    from pyproj import CRS, Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as stx
    d = json.load(open(path, encoding="utf-8"))
    src = ((d.get("crs") or {}).get("properties") or {}).get("name")
    geoms = [shape(f["geometry"]) for f in d.get("features", []) if f.get("geometry")]
    geoms = [g for g in geoms if not g.is_empty]
    if not geoms:
        return []
    if src is None:
        x = abs(geoms[0].bounds[0])
        src = "EPSG:4326" if x <= 180 else ("EPSG:3857" if x > 1_000_000 else crs)
    if CRS.from_user_input(src) != CRS.from_user_input(crs):
        tr = Transformer.from_crs(src, crs, always_xy=True)
        geoms = [stx(tr.transform, g) for g in geoms]
    return geoms


def extra_mask(region_dir, method, key, kind, shape, transform, crs):
    """Union mask of extra/*<method>[_key]_<kind>.geojson; None if no such file."""
    pats = [f"*_{method}_{key}_{kind}.geojson"] if key else []
    if not (key and method == "EtGr"):          # EtGr bounds are per map/profile
        pats.append(f"*_{method}_{kind}.geojson")
    for pat in pats:
        fs = glob.glob(os.path.join(region_dir, "extra", pat))
        if fs:
            geoms = read_geojson_polys(fs[0], crs.to_string() if hasattr(crs, "to_string") else crs)
            if not geoms:
                return None
            return rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, dtype=np.uint8)
    return None


def labels_for(L, region, method, key):
    sub = L[(L.region == region) & (L["mod"] == method)]
    if method == "MagGr":
        keyed = set(sub.key.unique()) - {""}
        if key in keyed:
            return sub[sub.key == key]
        return sub[sub.key == ""]
    return sub[sub.key == key]


def prep_one(path, L, out_dir=OUT, res=RES):
    region = os.path.relpath(path, DATA).split(os.sep)[0]
    rdir = os.path.join(DATA, region)
    method, key = raster_key(path)
    img, valid, t, crs = read_resampled(path, res, region_utm(rdir))
    H, W = valid.shape
    bounds = extra_mask(rdir, method, key, "границы", (H, W), t, crs)
    noise = extra_mask(rdir, method, key, "шум", (H, W), t, crs)
    v = valid.copy()
    if bounds is not None:
        v &= bounds.astype(bool)
    if noise is not None:
        v &= ~noise.astype(bool)
    lab = labels_for(L, region, method, key).to_crs(crs)
    objs = lab[lab.class_name == "object_poly"]
    obj = rasterize([(g, 1) for g in objs.geometry], out_shape=(H, W), transform=t, dtype=np.uint8) if len(objs) else np.zeros((H, W), np.uint8)
    pts = lab[lab.class_name == "finds_points"]
    inv = ~t
    finds = np.array([inv * (p.x, p.y) for p in pts.geometry], np.float32).reshape(-1, 2)
    name = f"{region}__{os.path.basename(path)[:-4]}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, name + ".npz"), img=img, obj=obj, valid=v.astype(np.uint8), finds=finds,
                        has_finds=np.array(method == "MagGr" and len(pts) > 0),
                        meta=json.dumps({"region": region, "method": method, "key": key, "crs": crs.to_string(),
                                         "transform": list(t)[:6], "src": os.path.relpath(path, DATA),
                                         "n_obj": int(len(objs)), "n_finds": int(len(pts))}, ensure_ascii=False))
    return name, method, key, H, W, len(objs), len(pts)


def all_rasters():
    return sorted(p for p in glob.glob(os.path.join(DATA, "*", "0[789]_*", "*.tif")) if raster_key(p)[0] in METHODS)


if __name__ == "__main__":
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    for p in all_rasters():
        print(*prep_one(p, L), flush=True)
