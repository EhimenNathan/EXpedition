"""Optical raster I/O shared by training prep and inference (SpOr / Ae / Or).

Scenes come in two shapes: small screenshots (~1000-1400 × 800-930 px, framed on a site, 0.1-3 m/px) and big
orthos (thousands of px at 0.06-0.5 m). Small scenes are used whole at native framing (long side → CHIP);
big scenes are resampled to TILE_RES m/px and cut into overlapping CHIP tiles.
"""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

CHIP = 1024
OVERLAP = 256
TILE_RES = 0.5
SMALL_PX = 4_000_000       # W*H below this → whole-scene chip


def to_rgb8(arr, alpha=None):
    """(C,H,W) any dtype → (H,W,3) uint8 and valid mask. 16-bit data gets a 1-99 % stretch over valid pixels."""
    C = arr.shape[0]
    rgb = arr[:3] if C >= 3 else np.repeat(arr[:1], 3, 0)
    if alpha is None and C == 4:
        alpha = arr[3]
    valid = (alpha > 0) if alpha is not None else np.ones(rgb.shape[1:], bool)
    valid &= rgb.sum(0) > 0
    if rgb.dtype != np.uint8:
        x = rgb.astype(np.float32)
        if valid.any():
            lo, hi = np.percentile(x[:, valid], [1, 99])
        else:
            lo, hi = x.min(), x.max()
        x = (x - lo) / max(hi - lo, 1e-6) * 255
        rgb = np.clip(x, 0, 255).astype(np.uint8)
    out = np.moveaxis(rgb, 0, -1).copy()
    out[~valid] = 0
    return out, valid


def is_small(src):
    return src.width * src.height < SMALL_PX


def read_small(src, chip=CHIP):
    """Whole scene resized so the long side = chip. Returns rgb, valid, transform of the resized grid."""
    s = chip / max(src.width, src.height)
    W, H = max(1, round(src.width * s)), max(1, round(src.height * s))
    arr = src.read(out_shape=(src.count, H, W), resampling=Resampling.bilinear if s < 1 else Resampling.cubic)
    rgb, valid = to_rgb8(arr)
    t = src.transform * Affine.scale(src.width / W, src.height / H)
    return rgb, valid, t


def big_grid(src, res=TILE_RES):
    """Output grid (W,H,transform) for a big scene resampled to `res` m/px (never upsampled beyond native ×2)."""
    mx, my = metres_per_unit(src)
    rx, ry = src.res[0] * mx, src.res[1] * my            # native ground sampling in metres
    r = max(res, min(rx, ry) / 2)
    W, H = max(1, int(round(src.width * rx / r))), max(1, int(round(src.height * ry / r)))
    return W, H, src.transform * Affine.scale(src.width / W, src.height / H)


def metres_per_unit(src):
    """Ground metres per CRS unit along x / y (geographic CRS: degrees at the scene centre; else 1)."""
    if src.crs is not None and src.crs.is_geographic:
        lat = np.radians((src.bounds.top + src.bounds.bottom) / 2)
        return 111320.0 * np.cos(lat), 110574.0
    return 1.0, 1.0


def read_window_resampled(src, x0, y0, w, h, W, H):
    """Window (x0,y0,w,h) of the resampled W×H grid, read from the source with on-the-fly resampling."""
    sx, sy = src.width / W, src.height / H
    win = Window(x0 * sx, y0 * sy, w * sx, h * sy)
    arr = src.read(window=win, out_shape=(src.count, h, w), boundless=True, fill_value=0,
                   resampling=Resampling.average if sx > 1.5 else Resampling.bilinear)
    return to_rgb8(arr)


def read_grid_raw(src, W, H, max_px=700_000_000):
    """The whole W×H resampled grid in ONE read (same resampling rule as read_window_resampled; ~10× faster than
    per-tile windows on big compressed sheets). None when the grid is too large → fall back to per-tile reads."""
    if W * H * src.count > max_px:
        return None
    sx = src.width / W
    return src.read(out_shape=(src.count, H, W), resampling=Resampling.average if sx > 1.5 else Resampling.bilinear)


def tile_from_grid(raw, x0, y0, w, h):
    """Tile of a read_grid_raw array, converted exactly like read_window_resampled."""
    return to_rgb8(np.ascontiguousarray(raw[:, y0:y0 + h, x0:x0 + w]))


def tiles(W, H, chip=CHIP, overlap=OVERLAP):
    step = chip - overlap
    xs = list(range(0, max(1, W - overlap), step)) or [0]
    ys = list(range(0, max(1, H - overlap), step)) or [0]
    for y in ys:
        for x in xs:
            yield x, y, min(chip, W - x), min(chip, H - y)
