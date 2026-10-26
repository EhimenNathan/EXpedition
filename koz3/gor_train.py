"""Hillfort (gorodishcha) localiser for optical scenes: segmentation of hillfort masks on whole scenes at 512 px.

Hillforts are 40-500 m shapes; peak heatmaps and box detectors find few of them (OOF ≤ 0.15 F1), and a blind
centre prior on small screenshots scores 0.26. Here the whole chip is resized to 512 px, a U-Net predicts the
hillfort mask, and the representative point of each confident component is emitted (it lies inside the shape).

  python -m koz3.gor_train --fold 0 ; python -m koz3.gor_train --oof
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3 import hm  # noqa: E402
from koz3.sp_train import CLS, load_chips, prep_dir, ROOT, MEAN, STD  # noqa: E402

S = 512
GOR = CLS.index("gorodishcha")


def chip_sets(mod="SpOr"):
    chips = load_chips(mod)
    use = [c for c in chips if c["kind"] == "small" or any(cl == GOR for cl, _ in c["objs"])]
    return use


def sample(c, rng, d):
    from rasterio.features import rasterize
    from shapely.affinity import scale as sscale
    im = cv2.imread(os.path.join(d, c["file"]))[:, :, ::-1]
    h, w = im.shape[:2]
    f = S / max(h, w)
    im = cv2.resize(im, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    pad = np.zeros((S, S, 3), np.uint8); pad[:im.shape[0], :im.shape[1]] = im
    gs = [sscale(g, f, f, origin=(0, 0)) for cl, g in c["objs"] if cl == GOR]
    m = rasterize([(g, 1) for g in gs if not g.is_empty], out_shape=(S, S), dtype=np.uint8) if gs else np.zeros((S, S), np.uint8)
    x = pad.astype(np.float32).transpose(2, 0, 1)
    v = (x.sum(0) > 0).astype(np.float32)
    if rng is not None:
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
        if rng.random() < 0.3:
            g = x.mean(0, keepdims=True); x = g + (x - g) * rng.uniform(0, 1.3)
        k = rng.integers(4); fl = rng.random() < 0.5
        x, m, v = [np.rot90(a, k, axes=(-2, -1)) for a in (x, m, v)]
        if fl:
            x, m, v = x[..., ::-1], m[..., ::-1], v[..., ::-1]
    x = (np.clip(x, 0, 255) - MEAN) / STD * v
    return np.ascontiguousarray(x, np.float32), np.ascontiguousarray(m, np.float32), np.ascontiguousarray(v, np.float32)


def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet("tu-convnext_tiny", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1,
                    decoder_channels=(256, 128, 64, 32, 16))


def train(fold, iters=3000, bs=4, lr=3e-4, seed=0):
    use = [c for c in chip_sets() if c["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed); d = os.path.join(prep_dir("SpOr"), "chips")
    pos = np.array([any(cl == GOR for cl, _ in c["objs"]) for c in use], float)
    p = np.where(pos > 0, 2.0, 1.0); p /= p.sum()
    model = build_model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    t0 = time.time()
    for i in range(iters):
        b = [sample(use[j], rng, d) for j in rng.choice(len(use), bs, p=p)]
        x, m, v = [torch.from_numpy(np.stack([q[k] for q in b])).cuda() for k in range(3)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        loss = hm.bce_dice(o, m[:, None], v[:, None])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if i % 500 == 0 or i == iters - 1:
            print(f"gor fold {fold} it {i} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(ROOT, "work", "spor", f"gor_f{fold}.pt"))


@torch.no_grad()
def predict(model, rgb):
    """rgb chip (H,W,3) → (prob map at S px, scale f): flip TTA."""
    h, w = rgb.shape[:2]; f = S / max(h, w)
    c = {"file": None, "objs": []}
    im = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    pad = np.zeros((S, S, 3), np.uint8); pad[:im.shape[0], :im.shape[1]] = im
    x = pad.astype(np.float32).transpose(2, 0, 1); v = (x.sum(0) > 0)
    x = torch.from_numpy(((x - MEAN) / STD * v).astype(np.float32))[None].cuda()
    acc = 0
    for fl in ((), (-1,), (-2,)):
        z = x.flip(fl) if fl else x
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = torch.sigmoid(model(z).float())
        acc = acc + (o.flip(fl) if fl else o)
    return (acc / 3)[0, 0].cpu().numpy() * v, f


def components(prob, thr=0.5, min_px=40):
    """→ list of (x, y, score) at S px: representative point of each mask component, score = its mean prob."""
    from shapely.geometry import Polygon
    m = (prob > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < min_px:
            continue
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cc = max(cs, key=cv2.contourArea)
        if len(cc) < 4:
            continue
        p = Polygon(cc[:, 0, :] + 0.5).buffer(0)
        if p.is_empty:
            continue
        rp = (max(p.geoms, key=lambda g: g.area) if p.geom_type != "Polygon" else p).representative_point()
        out.append((rp.x, rp.y, float(prob[lab == k].mean())))
    return out


def oof():
    import pandas as pd
    from pyproj import Transformer
    from rasterio.transform import Affine
    chips = load_chips("SpOr"); d = os.path.join(prep_dir("SpOr"), "chips")
    rows = []
    for fold in sorted({c["fold"] for c in chips}):
        model = build_model(False).cuda().eval()
        model.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(ROOT, "work", "spor", f"gor_f{fold}.pt")).items()})
        for c in [c for c in chips if c["fold"] == fold]:
            rgb = cv2.imread(os.path.join(d, c["file"]))[:, :, ::-1]
            prob, f = predict(model, rgb)
            t = Affine(*c["transform"]); tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
            for thr in (0.3, 0.5):
                for x, y, s in components(prob, thr):
                    X, Y = tr.transform(*(t * (x / f, y / f)))
                    rows.append((c["region"], c["kind"], thr, s, float(prob.max()), X, Y))
            X, Y = tr.transform(*(t * (c["w"] / 2, c["h"] / 2)))
            rows.append((c["region"], c["kind"], -1.0, float(prob.max()), float(prob.max()), X, Y))   # centre fallback
        print("gor oof fold", fold, flush=True)
    pd.DataFrame(rows, columns=["region", "kind", "thr", "score", "scene_max", "x", "y"]).to_parquet(
        os.path.join(ROOT, "work", "spor", "oof_gor_cands.parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--oof", action="store_true")
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    if a.fold is not None:
        train(a.fold, a.iters)
    if a.oof:
        oof()
