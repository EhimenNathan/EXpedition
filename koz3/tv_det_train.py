"""Satellite FCOS (torchvision, ResNet-50-FPN, COCO weights, BSD) on the previous team's 0.5 m scene prep and folds.

A detector family that trains on this machine (HF D-FINE / RT-DETR training does not — see docs/EXPERIMENTS.md).
OOF candidates (region, cls, score, x, y, fold) → work/spor/oof_fcos_cands.parquet for the nested fusion study.

  python -m koz3.tv_det_train --fold 0 ; python -m koz3.tv_det_train --oof
"""
import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
SP = os.path.join(ROOT, "kaggle_import/code/work/sp")
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]   # their order
RU = {"курганы_целые": 0, "курганы_поврежденные": 1, "городища": 2, "фортификации": 3, "архитектуры": 4}
CHIP = 640


def scenes():
    import shapely.wkb as wkb
    S = pd.read_csv(os.path.join(SP, "scenes.csv")).set_index("scene")
    L = pd.read_parquet(os.path.join(SP, "labels.parquet"))
    F = json.load(open(os.path.join(ROOT, "kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/folds.json")))
    out = {}
    from rasterio.transform import Affine
    for sc, r in S.iterrows():
        T = Affine(*json.loads(r["T"])); inv = ~T
        B = []
        for w, c in zip(L[L.scene == sc].wkb, L[L.scene == sc].cls):
            g = wkb.loads(w)
            x0, y0, x1, y1 = g.bounds
            (a, b), (c2, d) = inv * (x0, y1), inv * (x1, y0)
            B.append([RU[c], min(a, c2), min(b, d), max(a, c2), max(b, d)])
        out[sc] = dict(site=r["site"], fold=int(F.get(sc, -1)), T=T, crs=r["crs"], boxes=np.array(B, np.float32).reshape(-1, 5))
    return out


_IMG = {}


def img(sc):
    if sc not in _IMG:
        _IMG[sc] = np.load(os.path.join(SP, "img", sc + ".npy"), mmap_mode="r")
    return _IMG[sc]


def crop(SCN, train, rng):
    pools = [(s, i) for s in train for i in range(len(SCN[s]["boxes"]))]
    if pools and rng.random() < 0.6:
        s, i = pools[rng.integers(len(pools))]; b = SCN[s]["boxes"][i]
        cy, cx = (b[2] + b[4]) / 2 + rng.uniform(-CHIP / 3, CHIP / 3), (b[1] + b[3]) / 2 + rng.uniform(-CHIP / 3, CHIP / 3)
    else:
        s = train[rng.integers(len(train))]; H, W = img(s).shape[1:]; cy, cx = rng.uniform(0, H), rng.uniform(0, W)
    a = img(s); H, W = a.shape[1:]
    y0, x0 = int(cy - CHIP // 2), int(cx - CHIP // 2)
    ys, xs = slice(max(0, y0), min(H, y0 + CHIP)), slice(max(0, x0), min(W, x0 + CHIP))
    c = np.zeros((4, CHIP, CHIP), np.uint8)
    if ys.start < ys.stop and xs.start < xs.stop:
        c[:, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = a[:, ys, xs]
    B = SCN[s]["boxes"].copy(); B[:, [1, 3]] -= x0; B[:, [2, 4]] -= y0
    area = (B[:, 3] - B[:, 1]) * (B[:, 4] - B[:, 2]); Bc = B.copy(); Bc[:, 1:] = np.clip(Bc[:, 1:], 0, CHIP)
    keep = ((Bc[:, 3] - Bc[:, 1]) * (Bc[:, 4] - Bc[:, 2]) >= 0.5 * np.maximum(area, 1e-6)) & (Bc[:, 3] - Bc[:, 1] >= 2) & (Bc[:, 4] - Bc[:, 2] >= 2)
    B = Bc[keep]
    rgb = c[:3].astype(np.float32); vd = (c[3] > 0).astype(np.float32)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5, 7])); rgb = np.stack([cv2.GaussianBlur(ch, (k, k), 0) for ch in rgb])
    rgb = rgb * rng.uniform(0.75, 1.3, (3, 1, 1)) * rng.uniform(0.85, 1.15) + rng.uniform(-20, 20, (3, 1, 1))
    if rng.random() < 0.3:
        rgb = 255 * np.clip(rgb / 255, 0, 1) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.15:
        rgb = np.repeat(rgb.mean(0, keepdims=True), 3, 0)
    rgb = np.clip(rgb, 0, 255) * vd
    for _ in range(rng.integers(4)):
        rgb = np.rot90(rgb, 1, axes=(1, 2)); B = B[:, [0, 2, 1, 4, 3]].copy(); B[:, 2], B[:, 4] = CHIP - B[:, 4], CHIP - B[:, 2]
    if rng.random() < 0.5:
        rgb = rgb[:, :, ::-1]; B = B.copy(); B[:, 1], B[:, 3] = CHIP - B[:, 3], CHIP - B[:, 1]
    return np.ascontiguousarray(rgb / 255.0, np.float32), B


def build(pretrained=True):
    from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
    m = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1, weights_backbone=None) if pretrained else \
        fcos_resnet50_fpn(weights=None, weights_backbone=None)            # offline platform: never download
    from torchvision.models.detection.fcos import FCOSClassificationHead
    head = m.head.classification_head
    m.head.classification_head = FCOSClassificationHead(256, head.num_anchors, len(CLASSES) + 1)
    m.transform.min_size, m.transform.max_size = (CHIP,), CHIP
    m.score_thresh, m.detections_per_img, m.nms_thresh = 0.05, 300, 0.5
    return m


def train(fold, iters=8000, bs=8, lr=1e-4, seed=0, tag="fcos"):
    SCN = scenes(); tr = [s for s in SCN if SCN[s]["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    m = build().cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(4)
    nxt = lambda: [crop(SCN, tr, np.random.default_rng(rng.integers(1 << 31))) for _ in range(bs)]
    futs = [ex.submit(nxt) for _ in range(3)]; t0 = time.time(); run = None
    for it in range(iters):
        batch = futs.pop(0).result(); futs.append(ex.submit(nxt))
        X = [torch.from_numpy(x).cuda() for x, _ in batch]
        Y = [{"boxes": torch.from_numpy(B[:, 1:]).cuda(), "labels": torch.from_numpy(B[:, 0].astype(np.int64) + 1).cuda()} for _, B in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(m(X, Y).values())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0); scaler.step(opt); scaler.update(); sch.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if it % 500 == 0 or it == iters - 1:
            print(f"fcos fold {fold} it {it} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.join(ROOT, "work/spor"), exist_ok=True)
    torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(ROOT, f"work/spor/{tag}_f{fold}.pt"))


@torch.no_grad()
def detect_scene(m, a, T, ov=160, bs=4):
    import torchvision
    H, W = a.shape[1:]; step = CHIP - ov
    tiles = [(y, x) for y in range(0, max(H - ov, 1), step) for x in range(0, max(W - ov, 1), step)]
    Bs, Ss, Cs = [], [], []
    for k in range(0, len(tiles), bs):
        bt = tiles[k:k + bs]; X = []
        for y, x in bt:
            c = np.zeros((4, CHIP, CHIP), np.uint8); p = np.asarray(a[:, y:y + CHIP, x:x + CHIP]); c[:, :p.shape[1], :p.shape[2]] = p
            X.append(torch.from_numpy(c[:3].astype(np.float32) / 255.0 * (c[3] > 0)).cuda())
        for flip in (False, True):
            Xin = [t.flip(-1) for t in X] if flip else X
            with torch.autocast("cuda", dtype=torch.float16):
                outs = m(Xin)
            for (y, x), o in zip(bt, outs):
                b = o["boxes"].float().clone()
                if flip:
                    b[:, [0, 2]] = CHIP - b[:, [2, 0]]
                b[:, [0, 2]] += x; b[:, [1, 3]] += y
                Bs.append(b); Ss.append(o["scores"].float()); Cs.append(o["labels"] - 1)
    if not Bs:
        return []
    B, S, C = torch.cat(Bs), torch.cat(Ss), torch.cat(Cs)
    keep = torchvision.ops.batched_nms(B, S, C, 0.5)
    out = []
    for i in keep.tolist():
        x0, y0, x1, y1 = B[i].tolist(); X_, Y_ = T * ((x0 + x1) / 2, (y0 + y1) / 2)
        out.append((int(C[i]), X_, Y_, float(S[i])))
    return out


def oof(tag="fcos", folds=(0, 1, 2, 3)):
    from pyproj import Transformer
    SCN = scenes(); rows = []
    for fold in folds:
        m = build(False).cuda().eval()
        m.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(ROOT, f"work/spor/{tag}_f{fold}.pt")).items()})
        for sc, d in SCN.items():
            if d["fold"] != fold:
                continue
            tr = Transformer.from_crs(d["crs"], 3857, always_xy=True)
            for c, X, Y, s in detect_scene(m, img(sc), d["T"]):
                mx, my = tr.transform(X, Y)
                rows.append((d["site"], CLASSES[c], s, mx, my, fold))
        print("oof fold", fold, len(rows), flush=True)
    pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y", "fold"]).to_parquet(os.path.join(ROOT, f"work/spor/oof_{tag}_cands.parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--oof", action="store_true")
    ap.add_argument("--iters", type=int, default=8000); ap.add_argument("--tag", default="fcos"); ap.add_argument("--oof_folds", default="0,1,2,3")
    a = ap.parse_args()
    if a.fold is not None:
        train(a.fold, a.iters, tag=a.tag)
    if a.oof:
        oof(a.tag, tuple(int(f) for f in a.oof_folds.split(",")))
