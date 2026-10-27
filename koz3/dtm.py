"""LAS → DTM (shared by training prep and inference).

Ground = class 2 when it makes up ≥ 5 % of points; otherwise per-cell minimum + progressive morphological filter.
Points are streamed in chunks, so multi-GB clouds fit in memory. Cells without ground are filled from the nearest
ground cell and marked invalid if farther than FILL_M.
"""
import numpy as np
from scipy import ndimage as ndi

RES = 0.25
FILL_M = 3.0
MAX_CELLS = 250_000_000


def _bounds(path):
    import laspy
    with laspy.open(path) as r:
        h = r.header
        return h.mins, h.maxs, h.point_count


def ground_fraction(path, sample=2_000_000):
    import laspy
    with laspy.open(path) as r:
        for pts in r.chunk_iterator(sample):
            c = np.asarray(pts.classification)
            return float((c == 2).mean())
    return 0.0


def is_ground_only(path, sample=2_000_000, cell=1.0, max_spread=0.6):
    """True when the cloud is already ground-only (e.g. training site 101: all class 8). Within 1 m cells the vertical
    spread of a bare-earth cloud is small; vegetation / buildings give metres. PMF on such clouds shaves mounds."""
    import laspy
    with laspy.open(path) as r:
        for pts in r.chunk_iterator(sample):
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            k = ((x - x.min()) // cell).astype(np.int64) * 1_000_003 + ((y - y.min()) // cell).astype(np.int64)
            order = np.argsort(k); k, z = k[order], z[order]
            starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
            zmax = np.maximum.reduceat(z, starts); zmin = np.minimum.reduceat(z, starts)
            n = np.diff(np.r_[starts, len(k)])
            spread = (zmax - zmin)[n >= 3]
            return bool(spread.size) and float(np.percentile(spread, 95)) < max_spread
    return False


def build(path, res=RES, keep=1.0, seed=0):
    """→ dtm float32 (H,W) (NaN-free), valid bool (H,W), (x0, y_top, res) grid origin in the LAS CRS.

    keep < 1 randomly thins the cloud first (controlled density experiments, docs/PHYSICS_GATES_2026-10-27.md)."""
    import laspy
    mins, maxs, n = _bounds(path)
    while ((maxs[0] - mins[0]) / res) * ((maxs[1] - mins[1]) / res) > MAX_CELLS:
        res *= 2
    x0, y1 = np.floor(mins[0]), np.ceil(maxs[1])
    W = int(np.ceil((maxs[0] - x0) / res)) + 1
    H = int(np.ceil((y1 - mins[1]) / res)) + 1
    use_class = ground_fraction(path) >= 0.05
    all_ground = (not use_class) and is_ground_only(path)
    zsum = np.zeros(H * W, np.float64); cnt = np.zeros(H * W, np.int32)
    zmin = np.full(H * W, np.inf, np.float32)
    rng = np.random.default_rng(seed)
    with laspy.open(path) as r:
        for pts in r.chunk_iterator(5_000_000):
            if keep < 1.0:
                pts = pts[rng.random(len(pts)) < keep]
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z, np.float32)
            if use_class:
                m = np.asarray(pts.classification) == 2
                x, y, z = x[m], y[m], z[m]
            else:
                m = np.isin(np.asarray(pts.classification), (7, 18), invert=True)   # drop flagged noise
                x, y, z = x[m], y[m], z[m]
            ix = ((x - x0) / res).astype(np.int64); iy = ((y1 - y) / res).astype(np.int64)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            k = iy[ok] * W + ix[ok]
            if use_class or all_ground:
                zsum += np.bincount(k, z[ok], minlength=H * W); cnt += np.bincount(k, minlength=H * W).astype(np.int32)
            else:
                np.minimum.at(zmin, k, z[ok])
    if use_class or all_ground:
        has = cnt > 0
        g = np.where(has, zsum / np.maximum(cnt, 1), np.nan).astype(np.float32).reshape(H, W)
    else:
        g = np.where(np.isfinite(zmin), zmin, np.nan).reshape(H, W)
        g = pmf(g, res)
    has = np.isfinite(g)
    dist, (iy_, ix_) = ndi.distance_transform_edt(~has, return_indices=True)
    dtm = g[iy_, ix_]
    filled = ~has
    if filled.any():
        sm = ndi.gaussian_filter(dtm, 1.0)
        dtm = np.where(filled, sm, dtm)
    valid = dist * res <= FILL_M
    return dtm.astype(np.float32), valid, (float(x0), float(y1), float(res))


def pmf(zmin, res, slope=0.3, dh0=0.3, dh_max=3.0, windows_m=(1, 2, 4, 8, 16)):
    """Progressive morphological filter (Zhang et al. 2003) on a min-z grid; non-ground cells → NaN."""
    g = zmin.copy()
    fill = np.isnan(g)
    if fill.all():
        return g
    _, (iy, ix) = ndi.distance_transform_edt(fill, return_indices=True)
    surf = g[iy, ix]
    ground = ~fill
    prev = 0.0
    for w in windows_m:
        k = max(3, int(round(w / res)) | 1)
        opened = ndi.grey_opening(surf, size=(k, k))
        dh = min(dh_max, dh0 + slope * (w - prev)) if prev else dh0
        ground &= (surf - opened) <= dh
        surf = opened
        prev = w
    out = zmin.copy()
    out[~ground] = np.nan
    return out


def save(path_out, dtm, valid, origin, crs):
    zo = float(np.floor(dtm[valid].min())) if valid.any() else 0.0
    q = np.clip(np.round((dtm - zo) * 100), 0, 65535).astype(np.uint16)
    np.savez_compressed(path_out, dtm_cm=q, z0=zo, valid=valid, origin=np.array(origin), crs=str(crs))


def load(path):
    z = np.load(path)
    return z["dtm_cm"].astype(np.float32) / 100 + float(z["z0"]), z["valid"], tuple(z["origin"]), str(z["crs"])
