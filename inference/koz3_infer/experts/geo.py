"""Geophysics expert (MagGr / EtGr / TerFl): koz3.geo_train U-Net → finds peaks + object polygons."""
import json
import os

import numpy as np

from ..common import log, to_feature, circle
from ..models import cfg, ts_files, load_ts

FAMILIES = ["maggr", "etgr", "terfl"]
_MODELS = None


def READY():
    from ..models import model_dir
    return os.path.exists(os.path.join(model_dir("geo"), "objects.json"))


def _models():
    global _MODELS
    if _MODELS is None:
        _MODELS = [load_ts(w) for w in ts_files("geo")]
        log(f"geo: {len(_MODELS)} models")
    return _MODELS


def predict(region, paths, deadline):
    """object_poly: per map unit, 0.5 · CNN v1 (our 0.1 m grid) + 0.5 · pixel-LightGBM (previous team, reprojected),
    components ≥ thr (nested OOF 0.353). finds_points: physics candidates + LightGBM (predict_finds_gbm, 0.719)."""
    import json
    from pyproj import Transformer
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling
    from shapely.ops import transform as stx
    from koz3 import geo_data, geo_objects_core
    from koz3.gbm_np import NumpyGBM
    from koz3.geo_prep import read_resampled, raster_key, extra_mask, RES
    from koz3.geo_train import predict_map, objects_from_mask, geo_input
    from ..common import utm_crs
    from ..models import model_dir
    c = cfg()["geo"]
    oc = json.load(open(os.path.join(model_dir("geo"), "objects.json")))
    gbms = _obj_gbms(oc)
    feats = []
    for u in geo_data.discover(str(region.parent)):
        if u["region"] != region.name:
            continue
        method, key = raster_key(u["path"])
        img, valid, t, crs = read_resampled(u["path"], RES, utm_crs(region))
        H, W = valid.shape
        b = extra_mask(str(region), method, key, "границы", (H, W), t, crs)
        n = extra_mask(str(region), method, key, "шум", (H, W), t, crs)
        v = valid.copy()
        if b is not None: v &= b.astype(bool)
        if n is not None: v &= ~n.astype(bool)
        if v.sum() < 100:
            continue
        probs, ws = [], []
        if _models():
            x = geo_input(img, v, c.get("input", "rgb"))
            probs.append(np.mean([predict_map(m, x) for m in _models()], 0)[1]); ws.append(oc["w_cnn"])
        if gbms:
            d, pg = geo_objects_core.predict_prob(u, gbms, oc["features"])
            dst = np.zeros((H, W), np.float32)
            reproject(pg, dst, src_transform=d["T"], src_crs=d["crs"], dst_transform=t, dst_crs=crs, resampling=Resampling.bilinear)
            probs.append(dst); ws.append(1 - oc["w_cnn"])
        if not probs:
            continue
        prob = np.average(np.stack(probs), axis=0, weights=ws)
        local_res = RES
        etgr_cpu = _etgr_cpu()
        if method == "EtGr" and etgr_cpu is not None:
            from koz3 import geo_etgr_cpu
            a2, v2, t2, crs2 = geo_etgr_cpu.read_grid(u["path"], crs_fallback=utm_crs(region))
            H2, W2 = v2.shape
            b2 = extra_mask(str(region), method, key, "границы", (H2, W2), t2, crs2)
            n2 = extra_mask(str(region), method, key, "шум", (H2, W2), t2, crs2)
            if b2 is not None: v2 &= b2.astype(bool)
            if n2 is not None: v2 &= ~n2.astype(bool)
            if v2.sum() >= 100:
                cpu = geo_etgr_cpu.predict(a2, v2, etgr_cpu)
                deployed = np.zeros((H2, W2), np.float32)
                reproject(prob, deployed, src_transform=t, src_crs=crs,
                          dst_transform=t2, dst_crs=crs2, resampling=Resampling.bilinear)
                prob = 0.2 * cpu + 0.8 * deployed
                v, t, crs, local_res = v2, t2, crs2, geo_etgr_cpu.RES
        tr = Transformer.from_crs(crs, 3857, always_xy=True)
        f = lambda xx, yy: tr.transform(*(t * (np.asarray(xx, float), np.asarray(yy, float))))
        k = 0
        thr = oc.get("thr_method", {}).get(method, oc["thr"])     # per-method (nested: MagGr 0.75; EtGr / TerFl keep thr)
        min_area = 0.05 if (method == "EtGr" and local_res < RES) else oc["min_area_m2"]
        objs = objects_from_mask(prob, v, thr, int(min_area / local_res ** 2))
        if method == "EtGr" and local_res < RES:  # fixed 0.25 m interior patch used by the nested fusion gate
            from shapely.ops import polylabel
            from shapely.geometry import Point as _Pt
            objs = [(_Pt(q.x, q.y).buffer(0.25 / local_res, 12), sc_)
                    for p_, sc_ in objs for q in [polylabel(p_, tolerance=0.5)]]
        elif oc.get("anchor") == "pole":      # small interior patch at the pole of inaccessibility (nested 0.4197 → 0.4306)
            from shapely.ops import polylabel
            from shapely.geometry import Point as _Pt
            objs = [(_Pt(q.x, q.y).buffer(max(0.5, min(1.0 / RES, 0.8 * p_.exterior.distance(q))), 12), sc_)
                    for p_, sc_ in objs for q in [polylabel(p_, tolerance=0.5)]]
        for poly, sc in objs:
            ft = to_feature(stx(lambda xx, yy, z=None: f(xx, yy), poly), "object_poly", sc, region.name, method.lower())
            if ft:
                ft["properties"]["image"] = os.path.basename(u["path"])
                feats.append(ft); k += 1
        log(f"objects {region.name}/{method}/{key or '-'}: {k} polygons ({len(probs)} members)")
    if c.get("finds_source", "gbm") == "gbm":
        feats += predict_finds_gbm(region)
    return feats


_OBJ = None
_ETGR = None


def _etgr_cpu():
    global _ETGR
    if _ETGR is None:
        from koz3.gbm_np import NumpyGBM
        from ..models import model_dir
        p = os.path.join(model_dir("geo"), "etgr_cpu.json")
        _ETGR = NumpyGBM(p) if os.path.exists(p) else False
        log(f"geo EtGr CPU member: {bool(_ETGR)}")
    return _ETGR if _ETGR is not False else None


def _obj_gbms(oc):
    global _OBJ
    if _OBJ is None:
        from koz3.gbm_np import NumpyGBM
        from ..models import model_dir
        _OBJ = [NumpyGBM(os.path.join(model_dir("geo"), m)) for m in oc["models"]
                if os.path.exists(os.path.join(model_dir("geo"), m))]
        log(f"geo objects: {len(_OBJ)} GBM models")
    return _OBJ


_GBM = None


def predict_finds_gbm(region):
    """MagGr finds: physics candidates + dipole descriptors + LightGBM (numpy), previous team's v3 'base' recipe."""
    global _GBM
    import json
    from pyproj import Transformer
    from koz3 import geo_data, geo_finds_core
    from koz3.gbm_np import NumpyGBM
    from ..models import model_dir
    d = model_dir("geo")
    if _GBM is None:
        _GBM = (NumpyGBM(os.path.join(d, "finds_gbm.json")), json.load(open(os.path.join(d, "finds.json"), encoding="utf-8")))
    gbm, meta = _GBM
    feats = []
    for u in geo_data.discover(str(region.parent)):
        if u["region"] != region.name or u["method"] != "MagGr" or u["kind"] != "map":
            continue
        unit = geo_data.load_unit(u, meta["gsd"])
        df = geo_finds_core.candidates(unit)
        if not len(df):
            continue
        p = gbm.predict_proba(df[meta["features"]].values)[:, 1]
        keep = p >= meta["threshold"]
        tr = Transformer.from_crs(unit["crs"], 3857, always_xy=True)
        X, Y = tr.transform(df.x.values[keep], df.y.values[keep])
        for xx, yy, pp in zip(X, Y, p[keep]):
            k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(yy / 6378137.0)) - np.pi / 2)))
            ft = to_feature(circle(xx, yy, meta["pred_radius_m"] * k, 8), "finds_points", pp, region.name, "maggr")
            if ft: feats.append(ft)
        log(f"finds {region.name}/{u['key'] or 'grey'}: {len(df)} candidates → {int(keep.sum())} finds")
    return feats
