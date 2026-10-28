"""Satellite hillfort localiser (inference): DINOv3-SAT ViT-L/16 patch tokens of the whole scene at 512 and 768 px
(identity + horizontal flip), linear probes PCA(d)+logistic for d ∈ {32, 64, 128} × C ∈ {0.005, 0.05}, maps averaged on
the 768 grid, Gaussian σ 0.5 token, argmax → point. Trained by experiments/claude/sp_hillfort_dinov3_eval.py's recipe on
all training scenes (fit_final below); weights in models/hf_dino_loc.npz (~2 MB) + the DINOv3 backbone.

  from koz3.hf_dino_loc import Localiser
  loc = Localiser(backbone_dir_or_id, "models/hf_dino_loc.npz"); x3857, y3857, map_max = loc(path_to_tif)
"""
import os

import numpy as np

SIDES = (512, 768)
DIMS, CS = (32, 64, 128), (0.005, 0.05)
PATCH = 16


def read_scene(path, side):
    """Whole scene, longer side `side` px (multiple of 16), area-averaged → (uint8 HWC, affine of that image, crs).
    Identical to experiments/claude/sp_hillfort_dinov3_tokens.read_scene (the out-of-fold training path)."""
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
    if a.max() > 255:
        lo, hi = np.percentile(a, (1, 99)); a = (a - lo) / max(hi - lo, 1e-6) * 255
    return np.clip(a, 0, 255).astype(np.uint8).transpose(1, 2, 0), T, crs


def tokens(model, img, mean, std, device="cpu"):
    """img uint8 HWC (sides multiple of 16) → (gh·gw, D) tokens averaged over identity + horizontal flip."""
    import torch
    gh, gw = img.shape[0] // PATCH, img.shape[1] // PATCH
    skip = 1 + getattr(model.config, "num_register_tokens", 0)
    views = []
    for flip in (False, True):
        x = img[:, ::-1] if flip else img
        x = torch.from_numpy(((x.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy())[None].to(device)
        dt = torch.bfloat16 if device == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast(device, dtype=dt):
            t = model(pixel_values=x).last_hidden_state[0, skip:].float().cpu().numpy().reshape(gh, gw, -1)
        views.append(t[:, ::-1] if flip else t)
    return np.mean(views, 0).reshape(gh * gw, -1)


def probe_map(W, side, tok, gh, gw):
    """Mean probability over the six probes of one view → (gh, gw)."""
    out = []
    for d in DIMS:
        A = (tok - W[f"{side}_pca_mean"]) @ W[f"{side}_pca_comp"][:d].T
        for C in CS:
            k = f"{side}_{d}_{C}"
            z = ((A - W[k + "_mu"]) / W[k + "_sd"]) @ W[k + "_coef"] + W[k + "_b"]
            out.append(1 / (1 + np.exp(-z)))
    return np.mean(out, 0).reshape(gh, gw)


class Localiser:
    def __init__(self, backbone, weights, device="cpu", crs_override=None):
        """backbone: a local folder saved with save_backbone() (offline, fp16) or a Hugging Face id."""
        import torch
        from transformers import AutoModel, AutoImageProcessor
        tok = os.environ.get("HF_TOKEN")
        proc = AutoImageProcessor.from_pretrained(backbone, token=tok)
        self.mean = np.array(proc.image_mean, np.float32); self.std = np.array(proc.image_std, np.float32)
        self.model = AutoModel.from_pretrained(backbone, token=tok, dtype=torch.float32).to(device).eval()
        self.device = device; self.crs_override = crs_override
        self.W = dict(np.load(weights))

    def __call__(self, path, crs=None):
        """→ (x, y in EPSG:3857, map max). crs: the scene CRS to use when the raster's own CRS is unreliable (the
        deployment passes the same CRS it uses for the scene gate)."""
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter, zoom
        maps = []
        for side in SIDES:
            img, T, raster_crs = read_scene(path, side)
            gh, gw = img.shape[0] // PATCH, img.shape[1] // PATCH
            maps.append((probe_map(self.W, side, tokens(self.model, img, self.mean, self.std, self.device), gh, gw), T))
        img_crs = crs or self.crs_override
        (_, T), (gh, gw) = maps[-1], maps[-1][0].shape
        m = np.mean([zoom(a, (gh / a.shape[0], gw / a.shape[1]), order=1) if a.shape != (gh, gw) else a for a, _ in maps], 0)
        g = gaussian_filter(m, 0.5, mode="nearest"); r, c = np.unravel_index(int(np.argmax(g)), g.shape)
        X, Y = T * ((c + 0.5) * PATCH, (r + 0.5) * PATCH)
        x3, y3 = Transformer.from_crs(img_crs or raster_crs, 3857, always_xy=True).transform(X, Y)
        return float(x3), float(y3), float(g.max())


def save_backbone(model_id="facebook/dinov3-vitl16-pretrain-sat493m", out="models/dinov3_sat"):
    """Save the backbone + processor locally in fp16 for the offline archive (~0.6 GB)."""
    import torch
    from transformers import AutoModel, AutoImageProcessor
    tok = os.environ.get("HF_TOKEN")
    AutoImageProcessor.from_pretrained(model_id, token=tok).save_pretrained(out)
    AutoModel.from_pretrained(model_id, token=tok, dtype=torch.float16).save_pretrained(out)
    return out


def fit_final(token_files=("work/hf_dinov3/tokens.npz", "work/hf_dinov3/tokens768.npz"), out="models/hf_dino_loc.npz"):
    """Fit every probe on ALL training scenes (same recipe as the out-of-fold evaluation)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    W = {}
    for side, f in zip(SIDES, token_files):
        Z = np.load(f, allow_pickle=True); tok, y = Z["tok"].astype(np.float32), Z["y"]
        pca = PCA(max(DIMS), random_state=0).fit(tok[::4])
        W[f"{side}_pca_mean"] = pca.mean_.astype(np.float32); W[f"{side}_pca_comp"] = pca.components_.astype(np.float32)
        A = pca.transform(tok)
        for d in DIMS:
            sc = StandardScaler().fit(A[:, :d])
            for C in CS:
                m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(A[:, :d]), y)
                k = f"{side}_{d}_{C}"
                W[k + "_mu"], W[k + "_sd"] = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
                W[k + "_coef"], W[k + "_b"] = m.coef_[0].astype(np.float32), np.float32(m.intercept_[0])
    os.makedirs(os.path.dirname(out), exist_ok=True); np.savez(out, **W)
    return out
