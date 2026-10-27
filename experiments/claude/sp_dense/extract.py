"""Dense DINOv3-SAT features for every satellite scene at a fixed ground resolution (CPU, resumable).

Each scene is resampled to GSD metres per pixel (default 0.5 → 8 m tokens), cut into 512 px tiles with a 24-token
stride (4-token margins dropped, so every kept token has ≥ 64 m of context), and the kept tokens are stitched into one
scene-wide token grid. Tokens are projected to PCA_DIM dimensions with a PCA fitted once on a random sample of tiles from
all scenes (unsupervised: no labels involved) and stored as float16.

  python experiments/claude/sp_dense/extract.py pca      # fit the PCA on ~400 sample tiles → work/sp_dense/pca.npz
  python experiments/claude/sp_dense/extract.py run      # all scenes → work/sp_dense/scenes/<idx>.npz (skips done ones)
Per scene file: f (gh, gw, PCA_DIM) float16, T (6 affine coefficients of the token grid: token (r, c) centre =
T * (c + 0.5, r + 0.5)), crs (WKT), region, tif.
"""
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
MODEL = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-sat493m")
GSD = float(os.environ.get("SP_DENSE_GSD", 0.5))
OUT = os.path.join(ROOT, os.environ.get("SP_DENSE_OUT", "work/sp_dense"))
PATCH, TILE, STRIDE, MARGIN, PCA_DIM = 16, 32, 24, 4, 128


def scenes():
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    regs = set(L[L["mod"] == "SpOr"].region)
    tifs = sorted(glob.glob(os.path.join(ROOT, "data/train-expds-3/*/04_*/*.tif")))
    return [t for t in tifs if os.path.basename(os.path.dirname(os.path.dirname(t))) in regs]


def read_gsd(path, gsd=GSD):
    """Whole scene resampled to gsd m/px (bilinear), padded to a multiple of 16 px → uint8 HWC, affine, crs, valid HW."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as ds:
        res = abs(ds.transform.a)
        h, w = max(1, int(round(ds.height * res / gsd))), max(1, int(round(ds.width * res / gsd)))
        bands = list(range(1, min(3, ds.count) + 1))
        a = ds.read(indexes=bands, out_shape=(len(bands), h, w), resampling=Resampling.bilinear).astype(np.float32)
        T = ds.transform * ds.transform.scale(ds.width / w, ds.height / h); crs = ds.crs
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    if a.max() > 255:
        lo, hi = np.percentile(a, (1, 99)); a = (a - lo) / max(hi - lo, 1e-6) * 255
    a = np.clip(a, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    H = max(TILE * PATCH, int(np.ceil(h / PATCH)) * PATCH); W = max(TILE * PATCH, int(np.ceil(w / PATCH)) * PATCH)
    pad = np.empty((H, W, 3), np.uint8); pad[:] = a.reshape(-1, 3).mean(0).astype(np.uint8); pad[:h, :w] = a
    return pad, T, crs, (h, w)


def starts(G):
    s = list(range(0, max(G - TILE, 0) + 1, STRIDE))
    if s[-1] != G - TILE:
        s.append(G - TILE)
    return s


def keep_range(s, G):
    return (s + (0 if s == 0 else MARGIN), s + TILE - (0 if s == G - TILE else MARGIN))


class Backbone:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        torch.set_num_threads(os.cpu_count())
        proc = AutoImageProcessor.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
        self.mean = np.array(proc.image_mean, np.float32); self.std = np.array(proc.image_std, np.float32)
        self.m = AutoModel.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN")).eval()
        self.skip = 1 + getattr(self.m.config, "num_register_tokens", 0)

    def __call__(self, tiles):
        """list of uint8 (512, 512, 3) → (n, 32, 32, 1024) float32."""
        import torch
        x = np.stack([(t.astype(np.float32) / 255 - self.mean) / self.std for t in tiles]).transpose(0, 3, 1, 2)
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            h = self.m(pixel_values=torch.from_numpy(np.ascontiguousarray(x))).last_hidden_state[:, self.skip:]
        return h.float().numpy().reshape(len(tiles), TILE, TILE, -1)


def scene_tokens(bb, img, project, batch=4):
    gh, gw = img.shape[0] // PATCH, img.shape[1] // PATCH
    out = None; jobs = [(r, c) for r in starts(gh) for c in starts(gw)]
    for i in range(0, len(jobs), batch):
        part = jobs[i:i + batch]
        tok = bb([img[r * PATCH:(r + TILE) * PATCH, c * PATCH:(c + TILE) * PATCH] for r, c in part])
        for (r, c), t in zip(part, tok):
            t = project(t)
            if out is None:
                out = np.zeros((gh, gw, t.shape[-1]), np.float16)
            (r0, r1), (c0, c1) = keep_range(r, gh), keep_range(c, gw)
            out[r0:r1, c0:c1] = t[r0 - r:r1 - r, c0 - c:c1 - c]
    return out


def fit_pca(n_tiles=400, seed=0):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed); bb = Backbone(); S = scenes(); X = []
    per = max(1, n_tiles // len(S)) + 1
    for k, path in enumerate(S):
        img, _, _, _ = read_gsd(path)
        gh, gw = img.shape[0] // PATCH, img.shape[1] // PATCH
        jobs = [(r, c) for r in starts(gh) for c in starts(gw)]
        for j in rng.choice(len(jobs), min(per, len(jobs)), replace=False):
            r, c = jobs[j]
            X.append(bb([img[r * PATCH:(r + TILE) * PATCH, c * PATCH:(c + TILE) * PATCH]])[0].reshape(-1, 1024)[::4])
        print(f"pca sample {k + 1}/{len(S)}", flush=True)
    X = np.concatenate(X)
    p = PCA(PCA_DIM, random_state=0).fit(X)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "pca.npz"), mean=p.mean_.astype(np.float32), comp=p.components_.astype(np.float32),
             var=p.explained_variance_ratio_)
    print(f"PCA {PCA_DIM}: {len(X)} tokens, explained variance {p.explained_variance_ratio_.sum():.3f}")


def run():
    P = np.load(os.path.join(OUT, "pca.npz")); mean, comp = P["mean"], P["comp"]
    project = lambda t: (t - mean) @ comp.T
    bb = Backbone(); S = scenes(); os.makedirs(os.path.join(OUT, "scenes"), exist_ok=True); t0 = time.time()
    for k, path in enumerate(S):
        f = os.path.join(OUT, "scenes", f"{k:03d}.npz")
        if os.path.exists(f):
            continue
        img, T, crs, (h, w) = read_gsd(path)
        tok = scene_tokens(bb, img, project)
        Tt = T * T.scale(PATCH, PATCH)
        np.savez(f + ".tmp.npz", f=tok, T=np.array(Tt[:6]), crs=crs.to_wkt(), valid=np.array([h, w]) / PATCH,
                 region=os.path.basename(os.path.dirname(os.path.dirname(path))), tif=os.path.basename(path))
        os.replace(f + ".tmp.npz", f)
        print(f"{k + 1}/{len(S)} {os.path.basename(path)[:30]} grid {tok.shape[:2]} {time.time() - t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    {"pca": fit_pca, "run": run}[sys.argv[1]]()
