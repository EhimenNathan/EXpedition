"""Small CPU EtGr probability model used as a 20% member of the object ensemble.

The feature definition mirrors experiments/claude/geo_objects_cpu.py but uses a
NumPy CIE Lab conversion, because scikit-image is not available in the platform
image.  The model is a JSON LightGBM booster evaluated by koz3.gbm_np.
"""
import numpy as np
from scipy import ndimage as ndi


RES = 0.05


def rgb2lab(rgb):
    """sRGB [0, 1] -> CIE Lab (D65), matching skimage.color.rgb2lab."""
    x = np.asarray(rgb, np.float32)
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    X = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.95047
    Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.08883
    e, k = 0.008856, 903.3
    f = lambda q: np.where(q > e, np.cbrt(q), (k * q + 16.0) / 116.0)
    fx, fy, fz = f(X), f(Y), f(Z)
    return np.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), -1)


def read_grid(path, res=RES, crs_fallback=None):
    """Read a source raster on the experiment's exact ceil-sized grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(path) as ds:
        crs = ds.crs if ds.crs is not None else rasterio.crs.CRS.from_user_input(crs_fallback)
        b = ds.bounds
        W = int(np.ceil((b.right - b.left) / res)); H = int(np.ceil((b.top - b.bottom) / res))
        T = rasterio.transform.from_origin(b.left, b.top, res, res)
        nb = min(3, ds.count); a = np.zeros((nb, H, W), np.float32)
        reproject(ds.read(indexes=list(range(1, nb + 1))).astype(np.float32), a,
                  src_transform=ds.transform, src_crs=crs, dst_transform=T,
                  dst_crs=crs, resampling=Resampling.average)
        alpha = None
        if ds.count == 4:
            alpha = np.zeros((H, W), np.float32)
            reproject(ds.read(4).astype(np.float32), alpha, src_transform=ds.transform,
                      src_crs=crs, dst_transform=T, dst_crs=crs,
                      resampling=Resampling.average)
    valid = a.sum(0) > 0
    if alpha is not None:
        valid &= alpha > 127
    return a, valid, T, crs


def features(a, valid, res=RES, method="EtGr"):
    if a.shape[0] == 3:
        lab = rgb2lab(np.clip(a.transpose(1, 2, 0) / 255.0, 0, 1)).transpose(2, 0, 1)
        channels = list(lab) + [np.hypot(lab[1], lab[2])]
    else:
        channels = [a[0]]
    out = []
    for c in channels:
        v = c[valid]
        if not v.size:
            v = np.zeros(1)
        lo, hi = np.percentile(v, [2, 98])
        z = np.where(valid, (c - (lo + hi) / 2) / max(hi - lo, 1e-6), 0)
        out.append(z)
        for metres in (0.1, 0.3, 1.0, 3.0):
            s = max(metres / res, 0.5); smooth = ndi.gaussian_filter(z, s)
            out += [smooth, z - smooth,
                    np.sqrt(np.maximum(ndi.gaussian_filter(z * z, s) - smooth * smooth, 0))]
    methods = ("EtGr", "MagGr", "TerFl")
    mi = methods.index(method)
    out += [np.full(valid.shape, float(i == mi), np.float32) for i in range(3)]
    return np.stack(out).astype(np.float32)


def predict(a, valid, model, chunk=150_000):
    F = features(a, valid)
    rr, cc = np.nonzero(valid); p = np.zeros(valid.shape, np.float32)
    for start in range(0, len(rr), chunk):
        sl = slice(start, start + chunk)
        X = F[:, rr[sl], cc[sl]].T
        p[rr[sl], cc[sl]] = model.predict_proba(X)[:, 1]
    return p
