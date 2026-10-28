"""Phase 2 · E1: whole-scene patch tokens from other backbones for the hillfort localiser (same scenes, labels and file
format as experiments/claude/sp_hillfort_dinov3_tokens.py, so sp_hillfort_dinov3_eval.py can score / ensemble them).

  dinov2_b   timm vit_base_patch14_dinov2.lvd142m   (patch 14; web images)
  dinov2_s   timm vit_small_patch14_dinov2.lvd142m
  satlas     SatlasPretrain aerial Swin-V2-B, stage-3 features (stride 16, 512-d; 0.5–2 m aerial imagery)
Views: whole scene, longer side ≈ SIDE px (rounded to the patch), identity + horizontal flip averaged.
  python experiments/claude/p2/backbone_tokens.py <backbone> <side> <out.npz>
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "..", "sp_cnn"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(name, dev):
    """→ (callable x[N,3,H,W] (ImageNet-normalised) → tokens [N, gh, gw, D], patch size)."""
    import torch
    if name.startswith("dinov2"):
        import timm
        arch = {"dinov2_b": "vit_base_patch14_dinov2.lvd142m", "dinov2_s": "vit_small_patch14_dinov2.lvd142m"}[name]
        m = timm.create_model(arch, pretrained=True, num_classes=0, dynamic_img_size=True).to(dev).eval()
        npre = getattr(m, "num_prefix_tokens", 1)

        def f(x):
            t = m.forward_features(x)[:, npre:]
            gh, gw = x.shape[2] // 14, x.shape[3] // 14
            return t.reshape(len(x), gh, gw, -1)
        return f, 14
    if name == "satlas":
        import train as T
        enc = T.SatlasSwin.make(True).to(dev).eval()

        def f(x):
            return enc(x)[2].permute(0, 2, 3, 1)                 # stage 3: stride 16, 512 channels
        return f, 16
    raise ValueError(name)


def main(name, side, out):
    import geopandas as gpd
    import torch
    from pyproj import Transformer
    from shapely import contains_xy
    from tqdm.auto import tqdm
    from koz3.hf_dino_loc import read_scene
    side = int(side); dev = "cuda" if torch.cuda.is_available() else "cpu"
    f, P = load(name, dev)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]
    HF = {r: g.geometry.union_all() for r, g in L[L.class_name == "gorodishcha"].groupby("region")}
    tifs = sorted(glob.glob(os.path.join(ROOT, "data/train-expds-3/*/04_*/*.tif")))
    tifs = [t for t in tifs if os.path.basename(os.path.dirname(os.path.dirname(t))) in set(L.region)]
    Tok, Y, POS, SID, GRID, META = [], [], [], [], [], []
    for k, path in enumerate(tqdm(tifs, desc=f"{name} {side}px")):
        region = os.path.basename(os.path.dirname(os.path.dirname(path)))
        img, Taff, crs = read_scene(path, 16 * round(side / 16))       # 16-px multiple read, then resize to the patch grid
        import cv2
        h0, w0 = img.shape[:2]; s = side / max(h0, w0)
        gh, gw = max(1, round(h0 * s / P)), max(1, round(w0 * s / P))
        im = cv2.resize(img, (gw * P, gh * P), interpolation=cv2.INTER_AREA)
        Taff = Taff * Taff.scale(w0 / (gw * P), h0 / (gh * P))
        views = []
        for flip in (False, True):
            x = im[:, ::-1] if flip else im
            x = torch.from_numpy(((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())[None].to(dev)
            dt = torch.bfloat16 if dev == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode(), torch.autocast(dev, dtype=dt):
                t = f(x)[0].float().cpu().numpy()
            views.append(t[:, ::-1] if flip else t)
        tok = np.mean(views, 0).reshape(gh * gw, -1)
        rr, cc = np.mgrid[0:gh, 0:gw]
        X, Yp = Taff * ((cc.ravel() + 0.5) * P, (rr.ravel() + 0.5) * P)
        X3, Y3 = Transformer.from_crs(crs, 3857, always_xy=True).transform(np.asarray(X), np.asarray(Yp))
        y = contains_xy(HF[region], X3, Y3) if region in HF else np.zeros(gh * gw, bool)
        Tok.append(tok.astype(np.float16)); Y.append(y); SID.append(np.full(gh * gw, k)); GRID.append((gh, gw))
        POS.append(np.c_[(cc.ravel() + 0.5) / gw * 2 - 1, (rr.ravel() + 0.5) / gh * 2 - 1, X3, Y3])
        META.append((region, os.path.basename(path), 0.0, h0, w0))
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
    np.savez(os.path.join(ROOT, out), tok=np.concatenate(Tok), y=np.concatenate(Y), pos=np.concatenate(POS), sid=np.concatenate(SID),
             grid=np.array(GRID), meta=np.array(META, dtype=object))
    print("saved", out, sum(len(t) for t in Tok), "tokens")


if __name__ == "__main__":
    main(*sys.argv[1:])
