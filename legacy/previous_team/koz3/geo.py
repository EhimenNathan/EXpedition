"""Geophysics physics features: magnetometry dipole detection on the rendered grey map.
A small buried ferrous object produces a dipole: an adjacent positive and negative lobe whose axis depends on the local
inclination and the object's remanence. Rendering polarity and gain are unknown, so the detector is polarity- and
orientation-agnostic: oriented first-derivative-of-Gaussian energy (the dipole's antisymmetric signature) combined with
a scale-normalised LoG magnitude, evaluated at a few scales matched to finds of ~0.1-1 m."""
import numpy as np
from scipy import ndimage as ndi


def load_gray(path, target_gsd=0.1):
    """Read the rendered map, averaging down to target_gsd when finer (e.g. 3 cm plots)."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as r:
        f = max(1, int(round(target_gsd / abs(r.transform.a))))
        a = r.read(out_shape=(r.count, r.height // f, r.width // f), resampling=Resampling.average).astype(np.float32)
        T = r.transform * r.transform.scale(f, f); crs = r.crs
    g = a[:3].mean(0) if a.shape[0] >= 3 else a[0]
    valid = ~((a[:3] == 0).all(0) if a.shape[0] >= 3 else (a[0] == 0))
    if a.shape[0] == 4: valid &= a[3] > 0
    return g, valid, T, crs


def dipole_energy(g, valid, gsd, scales_m=(0.25, 0.5, 1.0), n_orient=8):
    x = g.copy(); x[~valid] = np.median(g[valid]) if valid.any() else 0
    x = (x - ndi.gaussian_filter(x, 5.0 / gsd))  # remove regional trend (5 m)
    x /= (np.std(x[valid]) + 1e-6)
    best = np.zeros_like(x)
    for s_m in scales_m:
        s = max(s_m / gsd, 0.7)
        gx = ndi.gaussian_filter(x, s, order=(0, 1)); gy = ndi.gaussian_filter(x, s, order=(1, 0))
        e = np.zeros_like(x)
        for k in range(n_orient):
            th = np.pi * k / n_orient
            e = np.maximum(e, np.abs(np.cos(th) * gx + np.sin(th) * gy))
        lap = np.abs(ndi.gaussian_laplace(x, s)) * s
        best = np.maximum(best, s * e + lap)
    best[~valid] = 0
    return best


def peaks(energy, gsd, min_dist_m=0.6, thr=None, q=0.995):
    from skimage.feature import peak_local_max
    t = thr if thr is not None else np.quantile(energy[energy > 0], q)
    return peak_local_max(energy, min_distance=max(1, int(min_dist_m / gsd)), threshold_abs=t)
