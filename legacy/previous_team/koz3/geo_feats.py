"""Per-pixel physics feature stack for rendered geophysics maps (magnetometry, resistivity, thermography).

Rendered 8-bit maps have unknown colour scale and gain, so everything starts from a robust per-map normalisation
(median / IQR inside the survey) and uses scale-space filters whose sizes are given in metres:
  * Gaussian-smoothed level and local contrast (difference from the 2 m and 5 m background),
  * signed scale-normalised Laplacian of Gaussian (blob detector; sign separates highs from lows),
  * oriented first-derivative energy (the antisymmetric signature of a magnetic dipole, any orientation),
  * gradient magnitude and local standard deviation (texture),
  * a dipole-pair score: strongest positive and strongest negative lobe within 1 m of each other.
All filters use NaN-aware normalised convolution so the survey edge does not create false features."""
import numpy as np
from scipy import ndimage as ndi

SCALES_M = (0.15, 0.3, 0.6, 1.2)


def robust_norm(x, valid):
    v = x[valid]
    if v.size < 10:
        return np.zeros_like(x)
    med = np.median(v); q1, q3 = np.percentile(v, [25, 75]); iqr = max(q3 - q1, 1e-3)
    z = (x - med) / iqr
    z[~valid] = 0
    return np.clip(z, -20, 20).astype(np.float32)


def nconv(x, valid, sigma_px):
    """Gaussian smoothing restricted to valid pixels (normalised convolution)."""
    w = valid.astype(np.float32)
    num = ndi.gaussian_filter(x * w, sigma_px); den = ndi.gaussian_filter(w, sigma_px)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)


def iter_channels(lum, valid, gsd, scales_m=SCALES_M, bg_m=(2.0, 5.0)):
    """Yield (name, HxW float32) feature channels one at a time (memory: a few full-size arrays at most).
    Finds use the default (fine) scales; objects pass coarser scales (e.g. 0.3-2.4 m) and background (5, 15 m)."""
    z = robust_norm(lum, valid)
    px = lambda m: max(m / gsd, 0.5)
    inv = ~valid
    def out(a):
        a = a.astype(np.float32, copy=False); a[inv] = 0; return a
    yield "z", z
    bg = nconv(z, valid, px(bg_m[0])); yield "hp2", out(z - bg)
    yield "absz", out(np.abs(z - bg)); del bg
    bg5 = nconv(z, valid, px(bg_m[1])); yield "hp5", out(z - bg5)
    zf = np.where(valid, z - bg5, 0).astype(np.float32)  # detrended field for derivative filters
    yield "bg5", out(bg5); del bg5
    k1 = int(max(3, round(1.0 / gsd)))
    for i, s in enumerate(scales_m):
        sp = px(s)
        yield f"g{s}", out(nconv(zf, valid, sp))
        lg = -ndi.gaussian_laplace(zf, sp) * sp * sp  # positive on bright blobs
        yield f"log{s}", out(lg.copy())
        if i in (1, 2):  # dipole pair score: bright and dark lobes within ~1 m
            yield f"dip{s}", out(np.sqrt(ndi.maximum_filter(np.maximum(lg, 0), k1) * ndi.maximum_filter(np.maximum(-lg, 0), k1)))
        del lg
        gx = ndi.gaussian_filter(zf, sp, order=(0, 1)); gy = ndi.gaussian_filter(zf, sp, order=(1, 0))
        yield f"grad{s}", out(np.hypot(gx, gy) * sp); del gx, gy
    for r in (0.5, 1.5):  # local texture
        m1 = nconv(zf, valid, px(r)); m2 = nconv(zf * zf, valid, px(r))
        yield f"std{r}", out(np.sqrt(np.maximum(m2 - m1 * m1, 0))); del m1, m2
    yield "edge_dist", (ndi.distance_transform_edt(valid) * gsd).astype(np.float32)


def stack(lum, valid, gsd, scales_m=SCALES_M, bg_m=(2.0, 5.0)):
    """Return (names, array [C, H, W]) of all channels (convenience wrapper; use iter_channels for big maps)."""
    names, arrs = [], []
    for n, a in iter_channels(lum, valid, gsd, scales_m, bg_m):
        names.append(n); arrs.append(a)
    return names, np.stack(arrs)


def candidate_score(names, A):
    """Untrained detector used to propose find candidates: blob strength (either sign) plus dipole pairing."""
    i = {n: k for k, n in enumerate(names)}
    s = np.zeros(A.shape[1:], np.float32)
    for sc in (0.15, 0.3, 0.6):
        s = np.maximum(s, np.abs(A[i[f"log{sc}"]]))
    return s + 0.5 * A[i["dip0.3"]]


def palette_scalar(rgb, valid):
    """Invert a rainbow ('jet'-like) colour scale to one scalar: dark blue -> blue -> cyan -> green -> yellow -> red ->
    dark red maps to roughly -0.3 ... 1.3. Rendered resistivity and thermography maps encode the measured value as
    hue, so luminance ranks colours wrongly (yellow is bright, red and blue are both dark). Returns (scalar, is_colour)
    where is_colour is False for grey maps (low saturation), in which case the caller should keep luminance."""
    import cv2
    x = np.clip(np.moveaxis(rgb, 0, -1), 0, 255).astype(np.uint8)
    h = cv2.cvtColor(x, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    hue = h[..., 0] * (360.0 / 255.0); sat = h[..., 1] / 255.0; val = h[..., 2] / 255.0
    is_colour = bool(np.median(sat[valid]) > 0.25) if valid.any() else False
    hh = np.where(hue > 300, hue - 360, hue)            # magenta-ish reds wrap to just below 0 deg
    t = np.clip((240.0 - hh) / 240.0, -0.1, 1.1)        # 240 deg (blue) -> 0, 0 deg (red) -> 1
    t = t + np.where(t > 0.85, 0.5 * (1 - val), 0) - np.where(t < 0.15, 0.5 * (1 - val), 0)  # darker ends extend the scale
    t[~valid] = 0
    return t.astype(np.float32), is_colour
