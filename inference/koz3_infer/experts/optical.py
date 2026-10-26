"""Optical experts (sp_or / ae / or): whole small scenes or resampled tiles → koz3.sp_train U-Net → peaks.

Instantiated per data type by sp_or.py / ae.py / or_.py (weights in models/<name>/, settings in config[<name>]).
"""
import os

import numpy as np

from ..common import log, utm_crs, oom_retry

_MODELS = {}
TILE_RES = {"sp_or": 0.5, "ae": 0.5, "or": 0.1}


def weights(name):
    from ..models import ts_files
    return ts_files(name)


def models(name):
    if name not in _MODELS:
        from ..models import load_ts
        _MODELS[name] = [load_ts(w) for w in weights(name)]
        log(f"{name}: {len(_MODELS[name])} models")
    return _MODELS[name]


def predict_scene(name, region, path, deadline, thr=None, dual=None):
    import time
    import rasterio
    from koz3 import sp_io, hm
    from koz3.sp_train import norm_img, decode_chip, dedupe, N_HM, N_MK
    from ..models import cfg
    c = cfg()[name]
    feats = []
    ms = models(name)
    with rasterio.open(path) as src:
        crs = src.crs.to_string() if (src.crs is not None and src.crs.to_epsg()) else utm_crs(region)
        if sp_io.is_small(src):
            rgb, valid, t = sp_io.read_small(src)
            chips = [(rgb, t)]
        else:
            W, H, tg = sp_io.big_grid(src, TILE_RES[name])
            chips = ((sp_io.read_window_resampled(src, x, y, w, h, W, H)[0], tg * rasterio.Affine.translation(x, y))
                     for x, y, w, h in sp_io.tiles(W, H))
        for rgb, t in chips:
            if time.time() > deadline:
                log(f"{name}: budget hit inside {path.name}")
                break
            x, v = norm_img(rgb)
            if v.mean() < 0.02:
                continue
            prob = np.mean([oom_retry(lambda t, m=m: hm.predict_tiled(m, x, N_HM + N_MK, tile=t, overlap=0 if t >= 1024 else 128)) for m in ms], 0)
            feats += decode_chip(prob, v, list(t)[:6], crs, region.name, name, c["thr"] if thr is None else thr, c["dual"] if dual is None else dual, c["r_out"])
    feats = dedupe(feats)
    for f in feats:
        f["properties"].update({"sub_region_name": "", "crs": "urn:ogc:def:crs:EPSG::3857"})
    return feats
