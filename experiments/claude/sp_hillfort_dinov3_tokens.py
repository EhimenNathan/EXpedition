"""Stage 1 of the satellite hillfort localiser: frozen DINOv3-SAT (ViT-L/16, SAT-493M) patch tokens for every satellite
scene, whole-scene view (longer side 512 px → 32 tokens), averaged over identity + horizontal-flip views, plus each
token's label (token centre inside a satellite hillfort polygon of the region), normalised position and scene metadata.

  python experiments/claude/sp_hillfort_dinov3_tokens.py [out=work/hf_dinov3/tokens.npz] [side=512]
CPU: ~3 s per view per scene at 512 px (4 cores).
"""
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
MODEL = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-sat493m")
PATCH = 16


def read_scene(path, side):
    """→ (uint8 HWC image with sides multiple of 16, longer side = side), rasterio transform of that image, crs."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as ds:
        s = side / max(ds.width, ds.height)
        h, w = max(PATCH, int(round(ds.height * s / PATCH)) * PATCH), max(PATCH, int(round(ds.width * s / PATCH)) * PATCH)
        bands = list(range(1, min(3, ds.count) + 1))
        a = ds.read(indexes=bands, out_shape=(len(bands), h, w), resampling=Resampling.average).astype(np.float32)
        T = ds.transform * ds.transform.scale(ds.width / w, ds.height / h); crs = ds.crs
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    if a.max() > 255:                                          # 16-bit: 1–99 % stretch
        lo, hi = np.percentile(a, (1, 99)); a = (a - lo) / max(hi - lo, 1e-6) * 255
    return np.clip(a, 0, 255).astype(np.uint8).transpose(1, 2, 0), T, crs


def main(out="work/hf_dinov3/tokens.npz", side=512):
    import geopandas as gpd
    import torch
    from pyproj import Transformer
    from shapely import contains_xy
    from transformers import AutoModel, AutoImageProcessor
    torch.set_num_threads(os.cpu_count())
    side = int(side)
    proc = AutoImageProcessor.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    mean = np.array(proc.image_mean, np.float32); std = np.array(proc.image_std, np.float32)
    model = AutoModel.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN")).eval()
    skip = 1 + getattr(model.config, "num_register_tokens", 0)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]
    HF = {r: g.geometry.union_all() for r, g in L[L.class_name == "gorodishcha"].groupby("region")}
    tifs = sorted(glob.glob(os.path.join(ROOT, "data/train-expds-3/*/04_*/*.tif")))
    tifs = [t for t in tifs if os.path.basename(os.path.dirname(os.path.dirname(t))) in set(L.region)]
    Tok, Y, POS, SID, GRID, META = [], [], [], [], [], []
    t0 = time.time()
    for k, path in enumerate(tifs):
        region = os.path.basename(os.path.dirname(os.path.dirname(path)))
        img, T, crs = read_scene(path, side)
        h, w = img.shape[:2]; gh, gw = h // PATCH, w // PATCH
        views = []
        for flip in (False, True):
            x = img[:, ::-1] if flip else img
            x = torch.from_numpy(((x.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy())[None]
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                t = model(pixel_values=x).last_hidden_state[0, skip:].float().numpy().reshape(gh, gw, -1)
            views.append(t[:, ::-1] if flip else t)
        tok = np.mean(views, 0).reshape(gh * gw, -1)
        rr, cc = np.mgrid[0:gh, 0:gw]
        X, Yp = T * ((cc.ravel() + 0.5) * PATCH, (rr.ravel() + 0.5) * PATCH)
        X3, Y3 = Transformer.from_crs(crs, 3857, always_xy=True).transform(np.asarray(X), np.asarray(Yp))
        y = contains_xy(HF[region], X3, Y3) if region in HF else np.zeros(gh * gw, bool)
        u = (cc.ravel() + 0.5) / gw * 2 - 1; v = (rr.ravel() + 0.5) / gh * 2 - 1        # −1…1 across the scene
        Tok.append(tok.astype(np.float16)); Y.append(y); POS.append(np.c_[u, v, X3, Y3]); SID.append(np.full(gh * gw, k))
        GRID.append((gh, gw)); META.append((region, os.path.basename(path), abs(T.a), h, w))
        print(f"{k + 1}/{len(tifs)} {region[:24]} {gh}x{gw} tokens, hillfort tokens {int(y.sum())}, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
    np.savez(os.path.join(ROOT, out), tok=np.concatenate(Tok), y=np.concatenate(Y), pos=np.concatenate(POS),
             sid=np.concatenate(SID), grid=np.array(GRID), meta=np.array(META, dtype=object))
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
