"""Ae (German WWII aerial film) detector trained on Ae itself: torchvision FCOS (ResNet-50-FPN, COCO) on the
data/ae_prep whole-sheet tiles (0.5 m, 1024 px). Folds = Ae's own 20 km site clusters (ann.json). OOF candidates of the
held-out sheets (region, cls, score, x, y in EPSG:3857, fold), duplicates from overlapping tiles merged within 4 m
→ work/ae/oof_{tag}_cands.parquet for the nested study (scripts/ae_det_oof.py).

  python -m koz3.ae_det_train --fold 0 ; python -m koz3.ae_det_train --oof
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3.sp_train import CLS, load_chips, prep_dir  # noqa: E402
from koz3.tv_det_train import build as build_fcos  # noqa: E402

CROP = 640
MOD = os.environ.get("DET_MOD", "Ae")
ARCH = os.environ.get("DET_ARCH", "fcos")          # fcos | dfine (HF D-FINE-L, no denoising queries — see EXPERIMENTS)
DET_CLASS = os.environ.get("DET_CLASS")             # optional single-class expert; keep original class id
DFINE = "ustc-community/dfine-large-coco"


class _HF(torch.nn.Module):
    """HF D-FINE behind the torchvision detection interface: list of (3,H,W) [0,1] images (+ targets) →
    loss dict in training, [{"boxes" xyxy px, "scores", "labels" 1-based}] in eval."""
    def __init__(self, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModelForObjectDetection
        kw = dict(num_labels=len(CLS), id2label=dict(enumerate(CLS)), label2id={c: i for i, c in enumerate(CLS)}, num_denoising=0)
        if pretrained:
            self.m = AutoModelForObjectDetection.from_pretrained(DFINE, ignore_mismatched_sizes=True, **kw)
        else:
            cfg = AutoConfig.from_pretrained(DFINE, **kw); cfg.use_pretrained_backbone = False
            self.m = AutoModelForObjectDetection.from_config(cfg)

    def forward(self, X, Y=None):
        x = torch.stack(X); n = x.shape[-1]
        if Y is not None:
            lab = []
            for y in Y:
                b = y["boxes"] / n
                lab.append({"class_labels": y["labels"] - 1, "boxes": torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)})
            return {"loss": self.m(pixel_values=x, labels=lab).loss}
        o = self.m(pixel_values=x); S = o.logits.float().sigmoid(); B = o.pred_boxes.float() * n
        out = []
        for s, b in zip(S, B):
            sc, cl = s.max(1); k = sc >= 0.05
            xy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
            out.append({"boxes": xy[k], "scores": sc[k], "labels": cl[k] + 1})
        return out


def build_det(pretrained=True):
    return _HF(pretrained) if ARCH == "dfine" else build_fcos(pretrained)


JOINT = os.environ.get("DET_JOINT") == "1"       # + satellite chips (grey) as extra kurgan sites, leakage-safe
QAUG = os.environ.get("DET_QAUG") == "1"         # + resolution loss / JPEG artefacts (degradation)
EXCL_M = 25_000.0                                  # satellite regions within 25 km of a held-out Ae site are excluded


def _centre3857(c):
    from pyproj import Transformer
    from rasterio.transform import Affine
    X, Y = Affine(*c["transform"]) * (c["w"] / 2, c["h"] / 2)
    return Transformer.from_crs(c["crs"], 3857, always_xy=True).transform(X, Y)


def train_chips(fold):
    """Ae chips of the other folds (+ grey SpOr chips farther than EXCL_M from every held-out Ae region)."""
    C = chips(); tr = [c for c in C if c["fold"] != fold]
    if not JOINT:
        return tr
    held = np.array([_centre3857(c) for c in C if c["fold"] == fold])
    sp = []
    for c in load_chips("SpOr"):
        if not c["boxes"]:
            continue
        x, y = _centre3857(c)
        if len(held) and np.min(np.hypot(held[:, 0] - x, held[:, 1] - y)) < EXCL_M:
            continue
        B = np.array([list(b[:4]) + [k] for b, k in zip(c["boxes"], c["classes"])], np.float32).reshape(-1, 5)
        sp.append(dict(c, B=B, src="SpOr"))
    print(f"joint: {len(tr)} Ae + {len(sp)} SpOr chips (SpOr within {EXCL_M / 1000:.0f} km of held-out Ae sites excluded)", flush=True)
    return tr + sp


def chips():
    out = []
    for c in load_chips(MOD):
        B = np.array([list(b[:4]) + [k] for b, k in zip(c["boxes"], c["classes"])], np.float32).reshape(-1, 5)
        if DET_CLASS:
            B = B[B[:, 4] == CLS.index(DET_CLASS)]
        out.append(dict(c, B=B))
    return out


_IM = {}


def image(c):
    if c["file"] not in _IM:
        _IM[c["file"]] = cv2.imread(os.path.join(prep_dir(c.get("src", MOD)), "chips", c["file"]))[:, :, ::-1].copy()
    return _IM[c["file"]]


def sample(tr, pos, rng):
    if pos and rng.random() < 0.7:
        if isinstance(pos, dict):                          # site-balanced: region first, then one of its boxes
            reg = list(pos)[rng.integers(len(pos))]; i, j = pos[reg][rng.integers(len(pos[reg]))]
        else:
            i, j = pos[rng.integers(len(pos))]
        c = tr[i]; b = c["B"][j]
        cx, cy = (b[0] + b[2]) / 2 + rng.uniform(-CROP / 3, CROP / 3), (b[1] + b[3]) / 2 + rng.uniform(-CROP / 3, CROP / 3)
    else:
        c = tr[rng.integers(len(tr))]; cx, cy = rng.uniform(0, c["w"]), rng.uniform(0, c["h"])
    im = image(c); H, W = im.shape[:2]
    x0, y0 = int(np.clip(cx - CROP / 2, 0, max(W - CROP, 0))), int(np.clip(cy - CROP / 2, 0, max(H - CROP, 0)))
    crop = np.zeros((CROP, CROP, 3), np.uint8); p = im[y0:y0 + CROP, x0:x0 + CROP]; crop[:p.shape[0], :p.shape[1]] = p
    B = c["B"].copy(); B[:, [0, 2]] -= x0; B[:, [1, 3]] -= y0
    area = np.maximum((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]), 1e-6); Bc = B.copy(); Bc[:, :4] = np.clip(Bc[:, :4], 0, CROP)
    keep = ((Bc[:, 2] - Bc[:, 0]) * (Bc[:, 3] - Bc[:, 1]) >= 0.5 * area) & (Bc[:, 2] - Bc[:, 0] >= 2) & (Bc[:, 3] - Bc[:, 1] >= 2)
    B = Bc[keep]
    # film photometry: grey, contrast / gamma / blur / grain; geometry: rot90 + flip
    g = crop.astype(np.float32).mean(2)
    g = g * rng.uniform(0.7, 1.35) + rng.uniform(-30, 30)
    if rng.random() < 0.4:
        g = 255 * np.clip(g / 255, 0, 1) ** rng.uniform(0.6, 1.6)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5])); g = cv2.GaussianBlur(g, (k, k), 0)
    if rng.random() < 0.3:
        g = g + rng.normal(0, rng.uniform(2, 10), g.shape)
    if QAUG and rng.random() < 0.4:                       # resolution loss (×1.5–3.5) then back
        f_ = rng.uniform(1.5, 3.5); m_ = max(8, int(CROP / f_))
        g = cv2.resize(cv2.resize(np.clip(g, 0, 255).astype(np.float32), (m_, m_), interpolation=cv2.INTER_AREA), (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    if QAUG and rng.random() < 0.35:                      # JPEG artefacts
        ok, enc = cv2.imencode(".jpg", np.clip(g, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 60))])
        g = cv2.imdecode(enc, 0).astype(np.float32)
    valid = crop.max(2) > 0
    x = np.repeat((np.clip(g, 0, 255) * valid)[None], 3, 0)
    for _ in range(rng.integers(4)):
        x = np.rot90(x, 1, axes=(1, 2)); B = B[:, [1, 0, 3, 2, 4]].copy(); B[:, 1], B[:, 3] = CROP - B[:, 3], CROP - B[:, 1]
    if rng.random() < 0.5:
        x = x[:, :, ::-1]; B = B.copy(); B[:, 0], B[:, 2] = CROP - B[:, 2], CROP - B[:, 0]
    return np.ascontiguousarray(x / 255.0, np.float32), B


def train(fold, iters=6000, bs=8, lr=1e-4, seed=0, tag="ae_fcos"):
    tr = train_chips(fold)
    pos = {}
    for i, c in enumerate(tr):
        for j in range(len(c["B"])):
            pos.setdefault(c["region"] + c.get("src", ""), []).append((i, j))
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    m = build_det().cuda().train()
    if ARCH == "dfine":             # their recipe: backbone lr × 0.1, grad clip 0.1
        bb = [p for n_, p in m.named_parameters() if "backbone" in n_]; oth = [p for n_, p in m.named_parameters() if "backbone" not in n_]
        opt = torch.optim.AdamW([{"params": bb, "lr": lr * 0.1}, {"params": oth, "lr": lr}], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=iters, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(4)
    nxt = lambda: [sample(tr, pos, np.random.default_rng(rng.integers(1 << 31))) for _ in range(bs)]
    futs = [ex.submit(nxt) for _ in range(3)]; t0 = time.time(); run = None
    print(f"{tag} fold {fold}: {len(tr)} train tiles, {sum(len(v) for v in pos.values())} boxes in {len(pos)} sites", flush=True)
    for it in range(iters):
        batch = futs.pop(0).result(); futs.append(ex.submit(nxt))
        X = [torch.from_numpy(x).cuda() for x, _ in batch]
        Y = [{"boxes": torch.from_numpy(B[:, :4]).cuda(), "labels": torch.from_numpy(B[:, 4].astype(np.int64) + 1).cuda()} for _, B in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(m(X, Y).values())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1 if ARCH == "dfine" else 10.0); scaler.step(opt); scaler.update(); sch.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if it % 500 == 0 or it == iters - 1:
            print(f"{tag} fold {fold} it {it} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.join(ROOT, "work/ae"), exist_ok=True)
    torch.save({k: v.half() for k, v in m.state_dict().items()}, os.path.join(ROOT, f"work/ae/{tag}_f{fold}.pt"))


@torch.no_grad()
def detect_image(m, im, ov=160, bs=4, fast=None):
    """im: HxWx3 uint8 (any size) → [(cls, cx, cy, w, h, score)] in pixels; grey input, class-wise NMS.
    Default: 640 px windows + flip TTA. fast (env DET_FAST=1): the whole image in ONE pass, no flip (~8× faster; FCOS is
    fully convolutional and the pixel scale is unchanged) — validated separately (OOF tag suffix _fast)."""
    import torchvision
    fast = os.environ.get("DET_FAST") == "1" if fast is None else fast
    g = im.astype(np.float32).mean(2); valid = im.max(2) > 0
    x_all = np.repeat((g * valid)[None], 3, 0) / 255.0
    if fast:
        H, W = g.shape
        X = torch.from_numpy(np.ascontiguousarray(x_all, np.float32)).cuda()
        m.transform.min_size, m.transform.max_size = (min(H, W),), max(H, W)          # no resize inside FCOS
        m.detections_per_img = 1000                                                   # dense cemeteries in one pass
        with torch.autocast("cuda", dtype=torch.float16):
            o = m([X])[0]
        B, S, Cl = o["boxes"].float(), o["scores"].float(), o["labels"] - 1
        keep = torchvision.ops.batched_nms(B, S, Cl, 0.5)
        return [(int(Cl[i]), float((B[i, 0] + B[i, 2]) / 2), float((B[i, 1] + B[i, 3]) / 2), float(B[i, 2] - B[i, 0]),
                 float(B[i, 3] - B[i, 1]), float(S[i])) for i in keep.tolist()]
    if hasattr(m, "transform"):
        m.transform.min_size, m.transform.max_size = (CROP,), CROP
    H, W = g.shape; step = CROP - ov
    tiles = [(y, x) for y in range(0, max(H - ov, 1), step) for x in range(0, max(W - ov, 1), step)]
    Bs, Ss, Cs = [], [], []
    for k in range(0, len(tiles), bs):
        bt = tiles[k:k + bs]; X = []
        for y, x in bt:
            c = np.zeros((3, CROP, CROP), np.float32); p = x_all[:, y:y + CROP, x:x + CROP]; c[:, :p.shape[1], :p.shape[2]] = p
            X.append(torch.from_numpy(c).cuda())
        for flip in (False, True):
            Xin = [t.flip(-1) for t in X] if flip else X
            with torch.autocast("cuda", dtype=torch.float16):
                outs = m(Xin)
            for (y, x), o in zip(bt, outs):
                b = o["boxes"].float().clone()
                if flip:
                    b[:, [0, 2]] = CROP - b[:, [2, 0]]
                b[:, [0, 2]] += x; b[:, [1, 3]] += y
                Bs.append(b); Ss.append(o["scores"].float()); Cs.append(o["labels"] - 1)
    if not Bs:
        return []
    B, S, Cl = torch.cat(Bs), torch.cat(Ss), torch.cat(Cs)
    keep = torchvision.ops.batched_nms(B, S, Cl, 0.5)
    return [(int(Cl[i]), float((B[i, 0] + B[i, 2]) / 2), float((B[i, 1] + B[i, 3]) / 2), float(B[i, 2] - B[i, 0]),
             float(B[i, 3] - B[i, 1]), float(S[i])) for i in keep.tolist()]


def load_model(path):
    m = build_det(False).cuda().eval()
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    return m


def oof(tag="ae_fcos", folds=(0, 1, 2, 3)):
    from pyproj import Transformer
    from rasterio.transform import Affine
    from koz3.sp_fuse import fuse
    C = chips(); rows = []
    for fold in folds:
        m = load_model(os.path.join(ROOT, f"work/ae/{tag}_f{fold}.pt"))
        for c in [c for c in C if c["fold"] == fold]:
            t = Affine(*c["transform"]); tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
            for cl, cx, cy, _, _, s in detect_image(m, image(c)):
                X, Y = tr.transform(*(t * (cx, cy)))
                rows.append((c["region"], c["scene"], CLS[cl], s, X, Y, fold))
        print("oof fold", fold, len(rows), flush=True)
    D = pd.DataFrame(rows, columns=["region", "scene", "cls", "score", "x", "y", "fold"])
    # A physical object repeated on two source images is two reference objects.  Keep
    # source images isolated while merging overlapping chips from the same image.
    D["unit"] = D["region"] + "\x1f" + D["scene"]
    Z = D.rename(columns={"region": "base_region", "unit": "region"})
    F = fuse([Z[["region", "cls", "score", "x", "y"]]], np.ones(1), match_m=4.0)
    parts = F.region.str.split("\x1f", n=1, expand=True)
    F["region"], F["scene"] = parts[0], parts[1]
    F["fold"] = F.region.map(D.groupby("region").fold.first())
    sfx = "_fast" if os.environ.get("DET_FAST") == "1" else ""
    layer = "_layers" if os.environ.get("DET_KEEP_SCENE") == "1" else ""
    if not layer:
        # Backward-compatible region-level file for older studies.
        F = fuse([F[["region", "cls", "score", "x", "y"]]], np.ones(1), match_m=4.0)
        F["fold"] = F.region.map(D.groupby("region").fold.first())
    F.to_parquet(os.path.join(ROOT, f"work/ae/oof_{tag}{sfx}{layer}_cands.parquet")); print(len(D), "raw →", len(F), "merged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--oof", action="store_true")
    ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--tag", default="ae_fcos"); ap.add_argument("--oof_folds", default="0,1,2,3")
    a = ap.parse_args()
    if a.fold is not None:
        train(a.fold, a.iters, tag=a.tag)
    if a.oof:
        oof(a.tag, tuple(int(f) for f in a.oof_folds.split(",")))
