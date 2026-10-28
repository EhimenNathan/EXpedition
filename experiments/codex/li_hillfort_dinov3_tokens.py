"""Frozen DINOv3-SAT tokens for whole-site LiDAR relief renders.

This reuses the model already packaged by v26.  Each prepared LiDAR site is
rendered from the deployed 25 m local-relief and two opposed hillshade channels,
then reduced to a 512-pixel whole-site view.  Output follows the satellite token
contract so the same geographic OOF localizer evaluation can be reused.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-sat493m")
LOCAL_MODEL = Path("/home/nathan/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-sat493m/snapshots/f692fa42da72c6797b67cd73494a168d1120d3ee")
SIDE = int(os.environ.get("LI_DINO_SIDE", "512"))
PATCH = 16


def resize_channel(a: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(np.asarray(a, np.float32), (w, h), interpolation=cv2.INTER_AREA)


def main(out="work/hf_dinov3/li_tokens512.npz"):
    import geopandas as gpd
    import torch
    from pyproj import Transformer
    from shapely import contains_xy
    from transformers import AutoImageProcessor, AutoModel
    from koz3.li_train import labels, site_items

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() and os.environ.get("LI_DINO_CPU") != "1" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    source = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else MODEL
    proc = AutoImageProcessor.from_pretrained(source, token=os.environ.get("HF_TOKEN"), local_files_only=LOCAL_MODEL.exists())
    mean = np.asarray(proc.image_mean, np.float32)
    std = np.asarray(proc.image_std, np.float32)
    model = AutoModel.from_pretrained(source, token=os.environ.get("HF_TOKEN"), torch_dtype=dtype,
                                      local_files_only=LOCAL_MODEL.exists()).to(device).eval()
    skip = 1 + getattr(model.config, "num_register_tokens", 0)
    L = labels()
    HF = {r: g.geometry.union_all() for r, g in L[L.class_name == "gorodishcha"].groupby("region")}
    items = site_items(L)
    tok_all, y_all, pos_all, sid_all, grids, meta = [], [], [], [], [], []
    t0 = time.time()
    for s, item in enumerate(items):
        region = item["region"]
        x = item["x"]
        H, W = x.shape[1:]
        scale = SIDE / max(H, W)
        h = max(PATCH, int(round(H * scale / PATCH)) * PATCH)
        w = max(PATCH, int(round(W * scale / PATCH)) * PATCH)
        # DINO receives a relief pseudo-RGB image: two opposed illumination
        # directions plus the broad local-relief channel.
        image = np.stack([
            (resize_channel(x[5], h, w) + 1) / 2,
            (resize_channel(x[6], h, w) + 1) / 2,
            resize_channel(x[3], h, w) / 8 + .5,
        ], 2)
        image = np.clip(image, 0, 1)
        views = []
        for flip in (False, True):
            z = image[:, ::-1].copy() if flip else image
            z = torch.from_numpy(((z - mean) / std).transpose(2, 0, 1).copy())[None].to(device)
            amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
            with torch.inference_mode(), torch.autocast(device, dtype=amp_dtype):
                q = model(pixel_values=z).last_hidden_state[0, skip:].float().numpy().reshape(h // PATCH, w // PATCH, -1)
            views.append(q[:, ::-1] if flip else q)
        tok = np.mean(views, 0).reshape(-1, views[0].shape[-1])
        gh, gw = h // PATCH, w // PATCH
        rr, cc = np.mgrid[0:gh, 0:gw]
        # Cache coordinates use top-left x/y and 0.5 m pixels.
        m = item["meta"]
        xp = m["x0"] + (cc.ravel() + .5) * PATCH * (W / w) * m["res"]
        yp = m["ytop"] - (rr.ravel() + .5) * PATCH * (H / h) * m["res"]
        tr = Transformer.from_crs(m["crs"], 3857, always_xy=True)
        X, Y = tr.transform(xp, yp)
        yy = contains_xy(HF[region], X, Y) if region in HF else np.zeros(len(X), bool)
        u = (cc.ravel() + .5) / gw * 2 - 1
        v = (rr.ravel() + .5) / gh * 2 - 1
        tok_all.append(tok.astype(np.float16)); y_all.append(yy)
        pos_all.append(np.c_[u, v, X, Y]); sid_all.append(np.full(len(X), s))
        grids.append((gh, gw)); meta.append((region, region + ".npz", m["res"] * max(H, W) / SIDE, h, w, item["fold"]))
        print(f"{s+1}/{len(items)} {region[:28]} {gh}x{gw} hillfort {int(yy.sum())} {time.time()-t0:.0f}s", flush=True)
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tok=np.concatenate(tok_all), y=np.concatenate(y_all), pos=np.concatenate(pos_all),
             sid=np.concatenate(sid_all), grid=np.asarray(grids), meta=np.asarray(meta, dtype=object))
    print("saved", path, path.stat().st_size)


if __name__ == "__main__":
    main(*sys.argv[1:])
