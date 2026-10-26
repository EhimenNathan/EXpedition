"""Hillfort-scene gate for small satellite screenshots: ConvNeXt-T (timm, ImageNet) classifies the whole scene
"contains a hillfort"; when confident, a hillfort point is emitted at the scene centre (screenshots are framed on the
site; the centre falls inside the hillfort in ~40 % of hillfort scenes). Folds = the shared satellite site folds.
OOF probabilities → work/spor/oof_scene_cls.parquet for the nested study.

  python -m koz3.scene_cls --fold 0 ; python -m koz3.scene_cls --oof
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3.sp_train import CLS, load_chips, prep_dir, ROOT, MEAN, STD  # noqa: E402

S = 384
GOR = CLS.index("gorodishcha")


def scenes():
    chips = [c for c in load_chips("SpOr") if c["kind"] == "small"]
    for c in chips:
        c["y"] = int(any(cl == GOR for cl, _ in c["objs"]))
    return chips


def load(c, rng=None):
    im = cv2.imread(os.path.join(prep_dir("SpOr"), "chips", c["file"]))[:, :, ::-1]
    h, w = im.shape[:2]; f = S / max(h, w)
    im = cv2.resize(im, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    pad = np.zeros((S, S, 3), np.uint8); oy, ox = (S - im.shape[0]) // 2, (S - im.shape[1]) // 2
    pad[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    x = pad.astype(np.float32).transpose(2, 0, 1)
    if rng is not None:
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
        if rng.random() < 0.3:
            g = x.mean(0, keepdims=True); x = g + (x - g) * rng.uniform(0, 1.3)
        x = np.rot90(x, rng.integers(4), axes=(1, 2))
        if rng.random() < 0.5:
            x = x[:, :, ::-1]
    return np.ascontiguousarray((np.clip(x, 0, 255) - MEAN) / STD, np.float32)


def build(pretrained=True):
    import timm
    return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=1, drop_path_rate=0.2)


def build_model(pretrained=False):          # scripts/build_submission.py export hook
    return build(pretrained)


def train(fold, iters=3000, bs=8, lr=1e-4, seed=0):
    ch = [c for c in scenes() if c["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    y = np.array([c["y"] for c in ch]); p = np.where(y == 1, 1.0 / max(y.sum(), 1), 1.0 / max((1 - y).sum(), 1)); p /= p.sum()
    m = build().cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.1)
    t0 = time.time()
    for it in range(iters):
        idx = rng.choice(len(ch), bs, p=p)
        x = torch.from_numpy(np.stack([load(ch[i], rng) for i in idx])).cuda()
        t = torch.tensor([ch[i]["y"] for i in idx], dtype=torch.float32).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x).float().squeeze(1), t)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if it % 300 == 0 or it == iters - 1:
            print(f"scene-cls fold {fold} it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(ROOT, f"work/spor/scene_cls_f{fold}.pt"))


@torch.no_grad()
def oof():
    from pyproj import Transformer
    from rasterio.transform import Affine
    rows = []
    for fold in range(4):
        m = build(False).cuda().eval()
        m.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(ROOT, f"work/spor/scene_cls_f{fold}.pt")).items()})
        for c in [c for c in scenes() if c["fold"] == fold]:
            x = torch.from_numpy(load(c))[None].cuda()
            ps = [torch.sigmoid(m(x.flip(fl) if fl else x)).item() for fl in ((), (-1,), (-2,))]
            t = Affine(*c["transform"]); X, Y = Transformer.from_crs(c["crs"], 3857, always_xy=True).transform(*(t * (c["w"] / 2, c["h"] / 2)))
            rows.append((c["region"], c["file"], fold, c["y"], float(np.mean(ps)), X, Y))
    D = pd.DataFrame(rows, columns=["region", "file", "fold", "y", "p", "x", "y3857"])
    D.to_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet"))
    from sklearn.metrics import roc_auc_score
    print(f"OOF scenes {len(D)} positives {int(D.y.sum())} AUC {roc_auc_score(D.y, D.p):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--oof", action="store_true")
    ap.add_argument("--iters", type=int, default=1500)
    a = ap.parse_args()
    if a.fold is not None:
        train(a.fold, a.iters)
    if a.oof:
        oof()
