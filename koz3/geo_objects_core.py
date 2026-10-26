"""(Ported from the previous team; models may be koz3.gbm_np.NumpyGBM.) Geophysics objects: per-pixel features, probability -> polygons. Shared by training (scripts/geo_objects_v3.py) and
inference (inference/run.py) so the deployed models see exactly the training features."""
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import rasterio.features as RF
from rasterio import features
from shapely.geometry import shape
from . import geo_data as GD, geo_feats as GF

GSD = {"MagGr": 0.2, "EtGr": 0.1, "TerFl": 0.3}; METHODS = list(GSD)
SC = (0.3, 0.6, 1.2, 2.4); PAL_SC = (0.6, 2.4)
PAL_KEEP = ("z", "hp5", "absz", "log0.6", "log2.4", "g2.4", "std1.5")


def hsv(rgb, valid):
    import cv2
    h = cv2.cvtColor(np.clip(np.moveaxis(rgb, 0, -1), 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32); h[~valid] = 0
    hue = h[..., 0] * (2 * np.pi / 180.0); s = h[..., 1] / 255
    return {"hue_sin": np.sin(hue) * s, "hue_cos": np.cos(hue) * s, "sat": s}


def stack_for(d, u, lum=None):
    """Feature stack (names, array [C,H,W]) and per-unit constants for a loaded unit d (koz3.geo_data.load_unit)."""
    v = d["valid"]
    names, A = GF.stack(d["lum"] if lum is None else lum, v, d["gsd"], scales_m=SC, bg_m=(5.0, 15.0))
    pal, is_col = GF.palette_scalar(d["rgb"], v)
    extra = hsv(d["rgb"], v)
    if is_col:
        pn, PA = GF.stack(pal * 255, v, d["gsd"], scales_m=PAL_SC, bg_m=(5.0, 15.0))
        for n, ch in zip(pn, PA):
            if n in PAL_KEEP:
                extra[f"pal_{n}"] = ch
    else:
        for n in PAL_KEEP:
            extra[f"pal_{n}"] = np.zeros_like(d["lum"])
    for k_, ch in extra.items():
        names.append(k_); A = np.concatenate([A, ch[None].astype(np.float32)])
    const = {f"m_{m}": float(u["method"] == m) for m in METHODS}
    const.update(is_profile=float(u["kind"] == "profile"), is_colour=float(is_col))
    return names, A, const


def to_frame(names, A, const, rr, cc):
    X = {n: A[i][rr, cc] for i, n in enumerate(names)}
    for k_, val in const.items():
        X[k_] = np.full(len(rr), val, np.float32)
    return pd.DataFrame(X)


def predict_prob(u, models, feats, chunk=150_000):
    """Probability map for one unit with a list of LightGBM boosters/classifiers (mean)."""
    d = GD.load_unit(u, GSD[u["method"]])
    names, A, const = stack_for(d, u)
    v = d["valid"]; rr, cc = np.nonzero(v); prob = np.zeros(v.shape, np.float32)
    for c0 in range(0, len(rr), chunk):
        sl = slice(c0, c0 + chunk)
        X = to_frame(names, A, const, rr[sl], cc[sl])[feats].to_numpy(np.float32)
        ps = [m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X) for m in models]
        prob[rr[sl], cc[sl]] = np.mean(ps, axis=0)
    return d, prob


def polygons(prob, valid, T, gsd, sigma, thr, min_area):
    m = (ndi.gaussian_filter(prob, sigma) >= thr) & valid if sigma > 0 else (prob >= thr) & valid
    m = ndi.binary_opening(m); lab, n = ndi.label(m)
    if n == 0:
        return []
    means = ndi.mean(prob, lab, np.arange(1, n + 1)); sizes = ndi.sum(np.ones_like(prob), lab, np.arange(1, n + 1))
    out = []
    for geom, val in RF.shapes(lab.astype(np.int32), mask=lab > 0, transform=T):
        i = int(val) - 1
        if sizes[i] * gsd * gsd < min_area:
            continue
        g = shape(geom).buffer(0)
        g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area)
        out.append((g.simplify(gsd, preserve_topology=True), float(means[i])))
    return out
