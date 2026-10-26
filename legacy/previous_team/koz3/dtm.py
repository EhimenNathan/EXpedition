"""LAS -> bare-earth DTM and archaeological relief channels.

Design notes
- The grid GSD adapts to point density (0.8-117 pts/m2 across sites); the result is
  resampled to a common model GSD so that object sizes in pixels are consistent.
- Holes are filled by normalised convolution (Gaussian-weighted mean of the
  observed cells at growing scales), which is exact where data exist.
- All relief channels are trend-removed or derivative quantities, so they are
  comparable across sites regardless of absolute elevation.
"""
import numpy as np
from scipy import ndimage as ndi

MODEL_GSD = 0.25  # metres per pixel for every LiDAR-derived training raster


def read_las(path, ground_only=True):
    import laspy
    las = laspy.read(path)
    cls = np.asarray(las.classification)
    keep = cls == 2 if (ground_only and (cls == 2).any()) else np.ones(len(cls), bool)
    x = np.asarray(las.x)[keep]; y = np.asarray(las.y)[keep]; z = np.asarray(las.z)[keep]
    it = np.asarray(las.intensity)[keep].astype(np.float32)
    return x, y, z, it, float(keep.mean())


def grid_res_for_density(dens):
    """Coarsest cell that still holds ~2 ground points on average."""
    for r in (0.25, 0.5, 1.0):
        if dens * r * r >= 1.0:
            return r
    return 1.0


def _normconv_fill(sum_, cnt, max_sigma_px=16):
    """Fill empty cells with a Gaussian-weighted mean of observed cells."""
    out = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan).astype(np.float32)
    hole = cnt == 0
    s = 1.0
    while hole.any() and s <= max_sigma_px:
        num = ndi.gaussian_filter(np.where(hole, 0, out).astype(np.float32), s)
        den = ndi.gaussian_filter((~hole).astype(np.float32), s)
        ok = hole & (den > 1e-3)
        out[ok] = num[ok] / den[ok]
        hole = hole & ~ok
        s *= 2
    return out, hole


def grid_dtm(x, y, z, it, bounds, res):
    """Mean-z binning on a north-up grid. bounds=(xmin,ymin,xmax,ymax). Returns dtm, intensity, count, transform."""
    xmin, ymin, xmax, ymax = bounds
    W = int(np.ceil((xmax - xmin) / res)); H = int(np.ceil((ymax - ymin) / res))
    c = ((x - xmin) / res).astype(np.int64); r = ((ymax - y) / res).astype(np.int64)
    m = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    idx = r[m] * W + c[m]
    cnt = np.bincount(idx, minlength=H * W).reshape(H, W).astype(np.float32)
    zs = np.bincount(idx, weights=z[m], minlength=H * W).reshape(H, W)
    its = np.bincount(idx, weights=it[m], minlength=H * W).reshape(H, W)
    dtm, hole_z = _normconv_fill(zs, cnt)
    inten, _ = _normconv_fill(its, cnt)
    transform = (xmin, res, 0.0, ymax, 0.0, -res)
    return dtm, inten, cnt, hole_z, transform


def resample_to(arr, res_from, res_to=MODEL_GSD, order=1):
    if abs(res_from - res_to) < 1e-9:
        return arr
    return ndi.zoom(arr, res_from / res_to, order=order).astype(np.float32)


# ---------------------------------------------------------------- relief channels
def gauss_m(a, sigma_m, gsd=MODEL_GSD):
    return ndi.gaussian_filter(a, sigma_m / gsd, mode="nearest")


def lrm(dtm, sigma_m, gsd=MODEL_GSD):
    """Local relief model: DTM minus a Gaussian trend surface (metres)."""
    return dtm - gauss_m(dtm, sigma_m, gsd)


def slope_deg(dtm, gsd=MODEL_GSD):
    gy, gx = np.gradient(dtm, gsd)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def hillshade(dtm, az=315.0, alt=35.0, gsd=MODEL_GSD):
    gy, gx = np.gradient(dtm, gsd)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1).astype(np.float32)


def log_blob(dtm, radius_m, gsd=MODEL_GSD):
    """Scale-normalised negative Laplacian of Gaussian: >0 on mounds of ~radius_m."""
    s = radius_m / np.sqrt(2) / gsd
    return (-(s ** 2) * ndi.gaussian_laplace(dtm, s)).astype(np.float32)


def openness(dtm, radius_m=10.0, n_dir=8, gsd=MODEL_GSD, positive=True):
    """Mean zenith (positive) / nadir (negative) horizon angle over n_dir directions (degrees).
    Low positive openness = enclosed (ditches, pits); low negative openness = exposed (mounds, ramparts)."""
    a = dtm if positive else -dtm
    steps = np.unique(np.geomspace(1, radius_m / gsd, 12).astype(int))
    H, W = a.shape
    acc = np.zeros_like(a, dtype=np.float32)
    for k in range(n_dir):
        th = 2 * np.pi * k / n_dir
        dx, dy = np.cos(th), -np.sin(th)
        maxang = np.full(a.shape, -np.inf, np.float32)
        for s in steps:
            ox, oy = int(round(dx * s)), int(round(dy * s))
            sh = np.roll(np.roll(a, -oy, 0), -ox, 1)
            d = np.hypot(ox, oy) * gsd
            maxang = np.maximum(maxang, np.arctan2(sh - a, d))
        acc += 90.0 - np.degrees(maxang)
    return acc / n_dir


def sky_view_factor(dtm, radius_m=10.0, n_dir=8, gsd=MODEL_GSD):
    steps = np.unique(np.geomspace(1, radius_m / gsd, 12).astype(int))
    acc = np.zeros_like(dtm, dtype=np.float32)
    for k in range(n_dir):
        th = 2 * np.pi * k / n_dir
        dx, dy = np.cos(th), -np.sin(th)
        maxang = np.zeros(dtm.shape, np.float32)
        for s in steps:
            ox, oy = int(round(dx * s)), int(round(dy * s))
            sh = np.roll(np.roll(dtm, -oy, 0), -ox, 1)
            maxang = np.maximum(maxang, np.arctan2(sh - dtm, np.hypot(ox, oy) * gsd))
        acc += 1.0 - np.sin(maxang)
    return acc / n_dir


def relief_stack(dtm, gsd=MODEL_GSD):
    """Channels used by the LiDAR expert (float32, all site-independent)."""
    ch = {
        "lrm5": lrm(dtm, 5.0, gsd), "lrm15": lrm(dtm, 15.0, gsd), "lrm40": lrm(dtm, 40.0, gsd),
        "slope": slope_deg(dtm, gsd),
        "log3": log_blob(dtm, 3.0, gsd), "log8": log_blob(dtm, 8.0, gsd), "log20": log_blob(dtm, 20.0, gsd),
        "svf": sky_view_factor(dtm, 10.0, 8, gsd),
        "opos": openness(dtm, 10.0, 8, gsd, True), "oneg": openness(dtm, 10.0, 8, gsd, False),
        "hs315": hillshade(dtm, 315, 35, gsd), "hs45": hillshade(dtm, 45, 35, gsd),
    }
    return ch


# ------------------------------------------------------------------ ground filter for LAS without class-2 points
def pmf(zmin, cell=1.0, s=0.15, dh0=0.3, dhmax=3.0, wins=(3, 5, 9, 17, 33)):
    """Progressive morphological filter (Zhang et al. 2003) on a minimum-elevation grid -> ground-cell mask."""
    ground = np.isfinite(zmin); prev = np.where(ground, zmin, np.nanmax(zmin)); w_prev = 1
    for w in wins:
        op = ndi.grey_opening(prev, size=(w, w))
        ground &= (prev - op) <= min(dh0 + s * (w - w_prev) * cell, dhmax)
        prev, w_prev = op, w
    return ground


def pmf_ground(las_path, xmin, ymax, H, W, cell=1.0):
    """Ground DTM + validity on the MODEL_GSD grid with origin (xmin, ymax) and shape (H, W), for unclassified LAS.
    Shared by training prep (scripts/prep_lidar_site.py) and inference (inference/run.py) so both see the same surface."""
    import laspy
    with laspy.open(las_path) as f:
        h = f.header; gx, gy = np.floor(h.mins[0]), np.ceil(h.maxs[1])
        Wc = int(np.ceil((np.ceil(h.maxs[0]) - gx) / cell)); Hc = int(np.ceil((gy - np.floor(h.mins[1])) / cell)); zmin = np.full(Hc * Wc, np.inf)
        for ch in f.chunk_iterator(10_000_000):
            c = ((np.asarray(ch.x) - gx) / cell).astype(np.int64); r = ((gy - np.asarray(ch.y)) / cell).astype(np.int64)
            m = (c >= 0) & (c < Wc) & (r >= 0) & (r < Hc)
            np.minimum.at(zmin, r[m] * Wc + c[m], np.asarray(ch.z)[m])
    zmin = zmin.reshape(Hc, Wc); zmin[~np.isfinite(zmin)] = np.nan
    g = pmf(zmin, cell); dtm, _ = _normconv_fill(np.where(g, np.nan_to_num(zmin), 0.0), g.astype(np.float32), max_sigma_px=16)
    gd = resample_to(np.nan_to_num(dtm, nan=float(np.nanmedian(dtm))), cell)
    gv = resample_to((ndi.distance_transform_edt(~g) * cell <= 6.0).astype(np.float32), cell, order=0) > 0.5
    ox, oy = int(round((xmin - gx) / MODEL_GSD)), int(round((gy - ymax) / MODEL_GSD))

    def fit(a, mode):  # shift the 1 m-filter grid onto the target grid, then pad/crop to (H, W)
        a = np.pad(a, ((max(-oy, 0), 0), (max(-ox, 0), 0)), mode=mode)[max(oy, 0):, max(ox, 0):]
        return np.pad(a, ((0, max(H - a.shape[0], 0)), (0, max(W - a.shape[1], 0))), mode=mode)[:H, :W]
    return fit(gd, "edge"), fit(gv, "constant"), float(g.mean())
