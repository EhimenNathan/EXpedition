"""Ae (1940s German aerial film) expert.

Our Ae-trained heatmap failed out-of-fold (honest 0.026: it fires everywhere on grey film), so Ae uses the satellite
D-FINE models (trained on SpOr only, greyscale-augmented) on the Ae sheet resampled to 0.5 m, keeping only damaged
kurgans at confidence ≥ 0.7: nested OOF on Ae labels pooled with the LiDAR + satellite experts 0.3085 → 0.3103
(the other classes lowered the pooled score). LiDAR twins (koz3.fuse) add co-located LiDAR detections."""
import numpy as np

from ..common import log, to_feature, circle
from ..models import cfg

FAMILIES = ["ae"]


def READY():
    from .sp_or import _det_weights
    return bool(_det_weights("sp_dfine")) or bool(_ae_weights())


def _ae_weights(name="ae_det"):
    import os
    from ..models import model_dir
    d = model_dir(name)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")) if os.path.isdir(d) else []


_AE = {}
_FORT_VERIFIER = None


def _fort_verifier_scores(crops):
    """Score 224 px greyscale crops with the frozen DINOv2-S verifier."""
    global _FORT_VERIFIER
    if not crops:
        return np.empty(0, np.float32)
    import os
    import torch
    from ..models import load_ts, model_dir
    if _FORT_VERIFIER is None:
        path = os.path.join(model_dir("ae_fort_verifier"), "model.ts")
        if not os.path.exists(path):
            return np.ones(len(crops), np.float32)
        _FORT_VERIFIER = load_ts(path)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(crops), 64):
            x = torch.from_numpy(np.stack(crops[i:i + 64])).cuda().float()[:, None].repeat(1, 3, 1, 1) / 255
            out.extend(_FORT_VERIFIER(((x - mean) / std).float()).float().cpu().numpy().tolist())
    return np.asarray(out, np.float32)


def _crop_fused_fortifications(F, chip_views, tr3857, half=112):
    """Attach each fused candidate to its highest-margin source chip, then crop.

    This is the same ordering and geometry as the geographic OOF verifier:
    tile detections -> per-image fusion -> best-margin chip -> 224 px crop.
    """
    import cv2
    from rasterio.transform import Affine
    forts = F[F.cls == "fortifikatsii"]
    crops, indices = [], []
    for idx, r in forts.iterrows():
        X, Y = tr3857.transform(float(r.x), float(r.y))
        best = None
        for gray, t in chip_views:
            px, py = (~t) * (X, Y)
            h, w = gray.shape
            margin = min(px, py, w - 1 - px, h - 1 - py)
            if margin >= half and (best is None or margin > best[0]):
                best = (margin, gray, int(round(px)), int(round(py)))
        if best is None:
            continue
        _, gray, px, py = best
        crops.append(gray[py-half:py+half, px-half:px+half])
        indices.append(idx)
    return indices, crops


def predict_ae_det(region, paths, deadline, c, name="ae_det", res=0.5, markup="ae"):
    """Ae-trained detector (koz3/ae_det_train.py): the sheet is cut exactly like data/ae_prep (koz3.sp_io: small scene
    whole, else 0.5 m grid in 1024 px tiles with 256 px overlap), detected per tile (grey, 640 px windows, flip TTA),
    tiles merged within 4 m (koz3.sp_fuse), per-class thresholds from the nested study (scripts/ae_det_oof.py)."""
    import time
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from koz3 import sp_io
    from koz3.ae_det_train import detect_image, load_model
    from koz3.sp_fuse import fuse
    from koz3.sp_train import CLS
    from ..common import utm_crs
    if name not in _AE:
        _AE[name] = [load_model(w) for w in _ae_weights(name)]
    thr = c["class_thr"]
    deadline = min(deadline, time.time() + c.get("max_s", 240))     # per-region cap: never starve the others
    rows = []
    for p in paths:
        t0 = time.time(); n0 = len(rows)
        chip_views = []
        with rasterio.open(str(p)) as src:
            crs = src.crs.to_string() if (src.crs is not None and src.crs.to_epsg()) else utm_crs(region)
            tr = Transformer.from_crs(crs, 3857, always_xy=True)
            tr3857 = Transformer.from_crs(3857, crs, always_xy=True)
            if sp_io.is_small(src):
                chips = [sp_io.read_small(src)]
            else:
                W, H, tg = sp_io.big_grid(src, res)
                try:
                    # The verifier was trained from independently resampled
                    # JPEG chips. Use that exact path when it is enabled.
                    raw = None if c.get("fort_verifier") else sp_io.read_grid_raw(src, W, H)
                except Exception:
                    raw = None
                chips = []
                for x, y, w, h in sp_io.tiles(W, H):
                    if time.time() > deadline:
                        log(f"{markup} {p.name}: deadline, {len(chips)} tiles"); break
                    try:
                        rgb, valid = sp_io.tile_from_grid(raw, x, y, w, h) if raw is not None else sp_io.read_window_resampled(src, x, y, w, h, W, H)
                    except rasterio.errors.RasterioIOError:
                        continue
                    if valid.mean() < 0.1:
                        continue
                    chips.append((rgb, valid, tg * rasterio.Affine.translation(x, y)))
            for rgb, valid, t in chips:
                if time.time() > deadline:
                    break
                # Training prep stores these exact chips as JPEG quality 92;
                # round-trip in memory so deployment crops match OOF pixels.
                import cv2
                ok, enc = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    chip_views.append((cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE), t))
                for m in _AE[name]:
                    for cl, cx, cy, _, _, s in detect_image(m, rgb, fast=c.get("fast", True)):
                        if CLS[cl] in thr and s >= thr[CLS[cl]]:
                            X, Y = tr.transform(*(t * (cx, cy)))
                            rows.append((region.name, CLS[cl], s, X, Y, p.name))
        # Fuse before visual verification. Filtering overlapping raw tile
        # detections first caused the rejected v21 rehearsal regression.
        if len(rows) > n0 and c.get("fort_verifier"):
            Di = pd.DataFrame(rows[n0:], columns=["region", "cls", "score", "x", "y", "image"])
            Fi = fuse([Di[["region", "cls", "score", "x", "y"]]], np.ones(1), match_m=4.0)
            idx, crops = _crop_fused_fortifications(Fi, chip_views, tr3857)
            vs = _fort_verifier_scores(crops)
            keep = np.ones(len(Fi), bool)
            lo = float(c["fort_verifier"].get("score_min", .35))
            vt = float(c["fort_verifier"].get("verifier_min", .20))
            fort_idx = set(Fi.index[Fi.cls == "fortifikatsii"])
            scored = dict(zip(idx, vs))
            for j in fort_idx:
                keep[j] = Fi.loc[j, "score"] >= lo and scored.get(j, -1.0) >= vt
            # Replace this image's raw rows with already-fused, verified rows;
            # the final per-image fusion below is then idempotent.
            rows[n0:] = [(region.name, r.cls, r.score, r.x, r.y, p.name)
                         for j, r in Fi[keep].iterrows()]
            log(f"{markup} {p.name}: verifier kept {int(keep.sum())}/{len(Fi)} fused detections")
        log(f"{markup} {p.name}: {len(chips)} tiles → {len(rows) - n0} detections {time.time() - t0:.0f}s")
    if not rows:
        return []
    feats = []
    D = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y", "image"])
    # Every aerial/orthophoto source image is an independent reference layer.
    # Fuse overlapping tiles within a source image, never between images.
    for image, Di in D.groupby("image", sort=False):
        F = fuse([Di[["region", "cls", "score", "x", "y"]]], np.ones(1), match_m=4.0)
        for r in F.itertuples():
            kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(r.y / 6378137.0)) - np.pi / 2)))
            ft = to_feature(circle(r.x, r.y, c["r_out"] * kk, 12), r.cls, r.score, region.name, markup)
            if ft:
                ft["properties"]["image"] = image
                ft["properties"]["source_layer"] = image
                feats.append(ft)
    return feats


def predict(region, paths, deadline, cfg_key="ae", markup="ae"):
    import time
    from pyproj import Transformer
    from koz3 import prev_sp_io, prev_sp_det
    from .sp_or import _det, PREV_CLASSES
    c = cfg()[cfg_key]
    feats = []
    det_name, res = {"ae": ("ae_det", 0.5), "or": ("or_det", 0.1)}[cfg_key]
    if c.get("class_thr") and _ae_weights(det_name):            # own-data FCOS (Ae fortifications / Or architecture)
        try:
            feats += predict_ae_det(region, paths, deadline, c, det_name, res, markup)
        except Exception as e:
            log(f"{markup} {region.name}: own detector failed {e!r}")
    keep = set(c.get("classes", ["kurgany_povrezhdennye"]))
    if not keep or not _det("sp_dfine"):
        return feats
    models = _det("sp_dfine")[: c.get("n_models", 2)]
    for p in paths:
        t0 = time.time()
        arr, T, crs, _ = prev_sp_io.read_scene(str(p), 0.5, max_px=1_600_000_000)   # large Ae sheets exceed the 120 Mpx default
        dets = prev_sp_det.detect_scene(models, arr, T, bs=4, time_left=lambda: deadline - time.time())
        tr = Transformer.from_crs(crs.to_string(), 3857, always_xy=True)
        n = 0
        for k, e, s in dets:
            cls = PREV_CLASSES[k]
            if cls not in keep or s < c["det_thr"]:
                continue
            x, y = tr.transform(e.centroid.x, e.centroid.y)
            kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
            ft = to_feature(circle(x, y, c["r_out"] * kk, 12), cls, s, region.name, markup)
            if ft:
                ft["properties"]["image"] = p.name
                ft["properties"]["source_layer"] = p.name
                feats.append(ft); n += 1
        log(f"{markup} {p.name}: {arr.shape[1]}x{arr.shape[2]} px @0.5 m, {len(dets)} detections → {n} kept {time.time() - t0:.0f}s")
    return feats
