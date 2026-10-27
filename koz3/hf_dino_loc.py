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


def tokens(model, img, mean, std, device="cpu"):
    """img uint8 HWC (sides multiple of 16) → (gh·gw, D) tokens averaged over identity + horizontal flip."""
    import torch
    gh, gw = img.shape[0] // PATCH, img.shape[1] // PATCH
    skip = 1 + getattr(model.config, "num_register_tokens", 0)
    views = []
    for flip in (False, True):
        x = img[:, ::-1] if flip else img
        x = torch.from_numpy(((x.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy())[None].to(device)
        with torch.inference_mode(), torch.autocast(device if device != "cuda" else "cuda", dtype=torch.bfloat16):
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
    def __init__(self, backbone, weights, device="cpu"):
        from transformers import AutoModel, AutoImageProcessor
        tok = os.environ.get("HF_TOKEN")
        proc = AutoImageProcessor.from_pretrained(backbone, token=tok)
        self.mean = np.array(proc.image_mean, np.float32); self.std = np.array(proc.image_std, np.float32)
        self.model = AutoModel.from_pretrained(backbone, token=tok).to(device).eval(); self.device = device
        self.W = dict(np.load(weights))

    def __call__(self, path):
        from pyproj import Transformer
        from scipy.ndimage import gaussian_filter, zoom
        from experiments.claude.sp_hillfort_dinov3_tokens import read_scene
        maps = []
        for side in SIDES:
            img, T, crs = read_scene(path, side)
            gh, gw = img.shape[0] // PATCH, img.shape[1] // PATCH
            maps.append((probe_map(self.W, side, tokens(self.model, img, self.mean, self.std, self.device), gh, gw), T))
        (_, T), (gh, gw) = maps[-1], maps[-1][0].shape
        m = np.mean([zoom(a, (gh / a.shape[0], gw / a.shape[1]), order=1) if a.shape != (gh, gw) else a for a, _ in maps], 0)
        g = gaussian_filter(m, 0.5, mode="nearest"); r, c = np.unravel_index(int(np.argmax(g)), g.shape)
        X, Y = T * ((c + 0.5) * PATCH, (r + 0.5) * PATCH)
        x3, y3 = Transformer.from_crs(crs, 3857, always_xy=True).transform(X, Y)
        return float(x3), float(y3), float(g.max())


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
