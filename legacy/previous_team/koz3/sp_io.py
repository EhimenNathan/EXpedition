"""Optical scene reader shared by training preparation and inference (satellite SpOr; reusable for Ae/Or).

read_scene(path, gsd) -> (uint8 array (4,H,W) = R,G,B,valid*255, Affine transform, CRS, info dict)
Reprojects to a metric UTM CRS at `gsd` metres: the file CRS if it is projected UTM, else the UTM zone of the scene
centre (EPSG:3857 / geographic / missing CRS). Area-average when downsampling, bilinear when upsampling."""
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling, ColorInterp
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin


def utm_for(lon, lat):
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def target_crs(src):
    c = src.crs
    if c is not None and c.is_projected and c.to_epsg() and 32601 <= c.to_epsg() <= 32760:
        return c
    l, b, r, t = transform_bounds(c or CRS.from_epsg(3857), "EPSG:4326", *src.bounds)
    return utm_for((l + r) / 2, (b + t) / 2)


def read_scene(path, gsd=0.5, max_px=120_000_000, bounds=None):
    """bounds: optional (l, b, r, t) in the scene's target UTM CRS (target_crs) to read only that window, e.g. the
    label extent + a margin on partially annotated aerial sheets."""
    with rasterio.open(path) as src:
        crs = target_crs(src)
        l, b, r, t = transform_bounds(src.crs or CRS.from_epsg(3857), crs, *src.bounds, densify_pts=21)
        if bounds is not None:
            l, b, r, t = max(l, bounds[0]), max(b, bounds[1]), min(r, bounds[2]), min(t, bounds[3])
            if r <= l or t <= b:
                raise ValueError("crop window does not overlap the scene")
        W = max(1, int(np.ceil((r - l) / gsd))); H = max(1, int(np.ceil((t - b) / gsd)))
        if W * H > max_px:
            raise ValueError(f"scene too large at {gsd} m: {W}x{H}")
        T = from_origin(l, t, gsd, gsd)
        native = abs(src.transform.a) if src.crs and src.crs.is_projected else None
        if native is None:  # approximate native GSD in metres
            native = (r - l) / src.width
        rs = Resampling.average if native < gsd else Resampling.bilinear
        ci = list(src.colorinterp); n = src.count
        alpha_idx = [i + 1 for i, c in enumerate(ci) if c == ColorInterp.alpha]
        order = []
        for want in (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
            order.append(ci.index(want) + 1 if want in ci else None)
        if None in order:  # no colour interpretation: assume the first three bands are R, G, B
            order = [1, 2, 3] if n >= 3 else [1, 1, 1]
        with WarpedVRT(src, crs=crs, transform=T, width=W, height=H, resampling=rs, nodata=src.nodata) as v:
            rgb = v.read(order).astype(np.float32)
            if alpha_idx:
                valid = v.read(alpha_idx[0]) > 0
            else:
                # warped areas outside the source footprint read as nodata/0
                valid = rgb.max(0) > 0
                if src.nodata is not None:
                    valid &= ~(rgb == src.nodata).all(0)
            # also mark pixels outside the source footprint (dataset mask)
            try:
                m = v.read_masks(1) > 0; valid &= m
            except Exception:
                pass
        info = dict(native_gsd=float(native), dtype=src.dtypes[0], bands=n)
    if info["dtype"] != "uint8":
        out = np.zeros_like(rgb)
        for i in range(3):
            vv = rgb[i][valid]
            lo, hi = (np.percentile(vv, [1, 99]) if vv.size else (0, 1))
            out[i] = (rgb[i] - lo) / max(hi - lo, 1e-6) * 255
        rgb = out
    arr = np.zeros((4, H, W), np.uint8)
    arr[:3] = np.clip(rgb, 0, 255).astype(np.uint8); arr[3] = valid.astype(np.uint8) * 255
    arr[:3, ~valid] = 0
    return arr, T, crs, info
