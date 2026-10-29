"""DINOv3-SAT ViT-L/16 satellite detector with a light multiscale spatial adapter.

This is the prospective foundation-backbone test for KOZ-3.  It deliberately
reuses the established satellite chips, labels, and geographic folds.  The
303M-parameter SAT-493M backbone is frozen; layer 8/16/24 patch tokens are
projected into a three-level feature pyramid and fused with a small convolutional
spatial prior.  A COCO-initialised FCOS head is then trained on KOZ-3.

The design follows the central DEIMv2 idea (a spatial tuning adapter turns
single-scale DINOv3 tokens into detector features) without depending on DEIMv2's
non-commercial source or its smaller web-pretrained backbone.

Examples:
  python -m koz3.dinov3_sat_det --fold 0 --iters 2500 --tag dsat
  python -m koz3.dinov3_sat_det --oof --tag dsat
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3.sp_train import CLS, load_chips, prep_dir, ROOT  # noqa: E402

SIDE = 512
DINO_LOCAL = os.environ.get(
    "DINOV3_SAT_LOCAL",
    "/home/nathan/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-sat493m/"
    "snapshots/f692fa42da72c6797b67cd73494a168d1120d3ee",
)
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)


class Sampler:
    def __init__(self, chips, rng):
        self.chips, self.rng = chips, rng
        self.dir = os.path.join(prep_dir("SpOr"), "chips")
        n = np.asarray([len(c["objs"]) for c in chips])
        self.p = (0.5 + np.sqrt(n)) * np.asarray([c["valid_frac"] for c in chips])
        self.p /= self.p.sum()
        self.cache = {}

    def image(self, name):
        im = self.cache.get(name)
        if im is None:
            im = cv2.imread(os.path.join(self.dir, name))[:, :, ::-1]
            if len(self.cache) < 800:
                self.cache[name] = im
        return im

    def crop(self):
        r = self.rng
        c = self.chips[r.choice(len(self.chips), p=self.p)]
        im = self.image(c["file"]); h, w = im.shape[:2]
        scale = float(np.exp(r.uniform(np.log(0.65), np.log(1.35))))
        src = SIDE / scale
        if c["objs"] and r.random() < 0.65:
            p = c["objs"][r.integers(len(c["objs"]))][1].representative_point()
            cx, cy = p.x + r.uniform(-src / 3, src / 3), p.y + r.uniform(-src / 3, src / 3)
        else:
            cx, cy = r.uniform(0, w), r.uniform(0, h)
        x0 = float(np.clip(cx - src / 2, min(0, w - src), max(0, w - src)))
        y0 = float(np.clip(cy - src / 2, min(0, h - src), max(0, h - src)))
        M = np.asarray([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]], np.float32)
        out = cv2.warpAffine(np.ascontiguousarray(im), M, (SIDE, SIDE), flags=cv2.INTER_LINEAR, borderValue=0)
        boxes, labels = [], []
        for cl, g in c["objs"]:
            gx0, gy0, gx1, gy1 = g.bounds
            b = np.asarray([(gx0-x0)*scale, (gy0-y0)*scale, (gx1-x0)*scale, (gy1-y0)*scale])
            area = max((b[2]-b[0])*(b[3]-b[1]), 1e-6); bc = np.clip(b, 0, SIDE)
            if (bc[2]-bc[0])*(bc[3]-bc[1]) >= 0.5*area and min(bc[2]-bc[0], bc[3]-bc[1]) >= 2:
                boxes.append(bc); labels.append(cl + 1)  # torchvision reserves 0 for background
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        labels = np.asarray(labels, np.int64)

        # Sensor/style robustness fixed before viewing any held-out result.
        x = out.astype(np.float32)
        x = x * r.uniform(0.78, 1.25, (1, 1, 3)) + r.uniform(-18, 18)
        if r.random() < 0.25:
            grey = x.mean(2, keepdims=True); x = grey + (x-grey)*r.uniform(0.2, 1.25)
        if r.random() < 0.15:
            k = int(r.choice([3, 5])); x = cv2.GaussianBlur(x, (k, k), 0)
        x = np.clip(x, 0, 255)
        k = int(r.integers(4))
        if k:
            x = np.rot90(x, k).copy()
            for _ in range(k):
                old = boxes.copy()
                boxes[:, 0], boxes[:, 1] = old[:, 1], SIDE-old[:, 2]
                boxes[:, 2], boxes[:, 3] = old[:, 3], SIDE-old[:, 0]
        if r.random() < 0.5:
            x = x[:, ::-1].copy(); boxes[:, 0], boxes[:, 2] = SIDE-boxes[:, 2].copy(), SIDE-boxes[:, 0].copy()
        return torch.from_numpy(x.transpose(2, 0, 1).copy() / 255.0).float(), {
            "boxes": torch.from_numpy(boxes.copy()).float(), "labels": torch.from_numpy(labels.copy()).long()
        }

    def batch(self, n):
        b = [self.crop() for _ in range(n)]
        return [z[0] for z in b], [z[1] for z in b]


class SpatialAdapter(nn.Module):
    """Frozen SAT ViT plus trainable 1/8, 1/16 and 1/32 feature pyramid."""
    out_channels = 256

    def __init__(self, tune_last=0):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained(DINO_LOCAL, dtype=torch.float16, local_files_only=True)
        self.dino.requires_grad_(False); self.dino.eval(); self.tune_last = int(tune_last)
        if self.tune_last:
            for block in self.dino.model.layer[-self.tune_last:]:
                block.float().requires_grad_(True)
            self.dino.norm.float().requires_grad_(True)
        self.register_buffer("mean", torch.tensor(SAT_MEAN)[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(SAT_STD)[None, :, None, None], persistent=False)
        self.proj = nn.ModuleList([nn.Conv2d(1024, 256, 1) for _ in range(3)])
        self.detail = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.GroupNorm(16, 128), nn.GELU(),
        )
        self.d8 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU())
        self.d16 = nn.Sequential(nn.Conv2d(256, 256, 3, 2, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU())
        self.d32 = nn.Sequential(nn.Conv2d(256, 256, 3, 2, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU())
        self.fuse = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.GroupNorm(32, 256), nn.GELU()) for _ in range(3)])

    def train(self, mode=True):
        super().train(mode); self.dino.eval(); return self

    def forward(self, x):
        z = (x - self.mean) / self.std
        # Backbone activations are constant; no gradient graph or activation stash.
        if self.tune_last:
            with torch.autocast("cuda", dtype=torch.float16):
                h = self.dino(pixel_values=z.half(), output_hidden_states=True).hidden_states
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = self.dino(pixel_values=z.half(), output_hidden_states=True).hidden_states
        b, _, H, W = x.shape; gh, gw = H // 16, W // 16
        sem = []
        for j, idx in enumerate((8, 16, 24)):
            t = h[idx][:, 5:].transpose(1, 2).reshape(b, 1024, gh, gw)
            sem.append(self.proj[j](t.float()))
        q4 = self.detail(x); q8 = self.d8(q4); q16 = self.d16(q8); q32 = self.d32(q16)
        p8 = self.fuse[0](F.interpolate(sem[0], size=q8.shape[-2:], mode="bilinear", align_corners=False) + q8)
        p16 = self.fuse[1](sem[1] + q16)
        p32 = self.fuse[2](F.avg_pool2d(sem[2], 2) + q32)
        return OrderedDict((str(i), p) for i, p in enumerate((p8, p16, p32)))


def make_model(pretrained_head=True, tune_last=0):
    from torchvision.models.detection import FCOS
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    anchors = AnchorGenerator(((8,), (16,), (32,)), ((1.0,),) * 3)
    m = FCOS(SpatialAdapter(tune_last), num_classes=len(CLS)+1, min_size=SIDE, max_size=SIDE,
             image_mean=[0, 0, 0], image_std=[1, 1, 1], score_thresh=0.04,
             nms_thresh=0.5, detections_per_img=300, topk_candidates=1500,
             anchor_generator=anchors)
    if pretrained_head:
        from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
        src = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1).state_dict()
        dst = m.state_dict(); copied = {}
        for k, v in src.items():
            if k.startswith("head.") and "classification_head.cls_logits" not in k and k in dst and dst[k].shape == v.shape:
                copied[k] = v
        m.load_state_dict(copied, strict=False)
        print("COCO head tensors copied", len(copied), flush=True)
    return m


def train(fold, tag="dsat", iters=2500, bs=2, lr=2e-4, seed=2601, init=None, tune_last=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    S = Sampler([c for c in load_chips("SpOr") if c["fold"] != fold], rng)
    m = make_model(tune_last=tune_last)
    if init:
        m.load_state_dict(torch.load(init, map_location="cpu"), strict=False)
    m = m.cuda().train()
    bb = [p for n,p in m.named_parameters() if p.requires_grad and "backbone.dino." in n]
    other = [p for n,p in m.named_parameters() if p.requires_grad and "backbone.dino." not in n]
    params = bb + other
    print("trainable M", sum(p.numel() for p in params)/1e6, flush=True)
    groups = ([{"params": bb, "lr": lr*.05}, {"params": other, "lr": lr}] if bb else [{"params":other,"lr":lr}])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=([lr*.05,lr] if bb else [lr]), total_steps=iters, pct_start=0.08)
    scaler = torch.amp.GradScaler()
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(4); fut = [ex.submit(S.batch, bs) for _ in range(3)]
    t0 = time.time(); run = None
    for it in range(iters):
        xs, ys = fut.pop(0).result(); fut.append(ex.submit(S.batch, bs))
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [{k: v.cuda(non_blocking=True) for k, v in y.items()} for y in ys]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            losses = m(xs, ys); loss = sum(losses.values())
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 2.0); scaler.step(opt); scaler.update(); sch.step()
        run = float(loss) if run is None else 0.98*run + 0.02*float(loss)
        if it % 100 == 0 or it == iters-1:
            print(f"dsat fold {fold} it {it}/{iters} loss {run:.4f} sec {time.time()-t0:.0f} peakGB {torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
    out = os.path.join(ROOT, "work/spor", f"det_{tag}_f{fold}.pt")
    trainable_names={n for n,p in m.named_parameters() if p.requires_grad}
    keep = {k: v.half().cpu() for k, v in m.state_dict().items()
            if not k.startswith("backbone.dino.") or k in trainable_names}
    torch.save(keep, out); print("saved", out, os.path.getsize(out)/2**20, "MiB", flush=True)


def load_model(path):
    m = make_model(pretrained_head=False)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    return m.cuda().eval()


@torch.inference_mode()
def detect(m, rgb, overlap=128):
    import torchvision
    H, W = rgb.shape[:2]; step = SIDE-overlap
    ys = list(range(0, max(H-SIDE, 0)+1, step)); xs = list(range(0, max(W-SIDE, 0)+1, step))
    if not ys or ys[-1] != max(H-SIDE, 0): ys.append(max(H-SIDE, 0))
    if not xs or xs[-1] != max(W-SIDE, 0): xs.append(max(W-SIDE, 0))
    BB, SS, LL = [], [], []
    for y0 in ys:
        for x0 in xs:
            a = np.zeros((SIDE, SIDE, 3), np.uint8); p = rgb[y0:y0+SIDE, x0:x0+SIDE]; a[:p.shape[0], :p.shape[1]] = p
            z = torch.from_numpy(a.transpose(2,0,1).copy()).float().cuda()/255
            with torch.autocast("cuda", dtype=torch.float16): out = m([z])[0]
            if len(out["scores"]):
                b = out["boxes"].clone(); b[:, [0,2]] += x0; b[:, [1,3]] += y0
                BB.append(b); SS.append(out["scores"]); LL.append(out["labels"])
    if not BB: return []
    b, s, l = torch.cat(BB), torch.cat(SS), torch.cat(LL)
    keep = torchvision.ops.batched_nms(b, s, l, 0.45)
    ans = []
    for i in keep.tolist():
        if not (1 <= int(l[i]) <= len(CLS)): continue
        x0,y0,x1,y1 = b[i].tolist(); ans.append((int(l[i])-1,(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0,float(s[i])))
    return ans


def oof(tag="dsat", folds=None):
    import pandas as pd
    from pyproj import Transformer
    from rasterio.transform import Affine
    chips = load_chips("SpOr"); folds = sorted(set(c["fold"] for c in chips)) if folds is None else folds
    rows = []; d = os.path.join(prep_dir("SpOr"), "chips")
    for fold in folds:
        m = load_model(os.path.join(ROOT, "work/spor", f"det_{tag}_f{fold}.pt"))
        for c in [z for z in chips if z["fold"] == fold]:
            rgb = cv2.imread(os.path.join(d, c["file"]))[:, :, ::-1]
            aff = Affine(*c["transform"]); tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
            for cl,cx,cy,bw,bh,s in detect(m,rgb):
                X,Y = tr.transform(*(aff*(cx,cy))); rows.append((c["region"],CLS[cl],s,X,Y,bw,bh,fold))
        print("oof fold",fold,"rows",len(rows),flush=True); del m; torch.cuda.empty_cache()
    f = os.path.join(ROOT,"work/spor",f"oof_det_{tag}_cands.parquet")
    pd.DataFrame(rows,columns=["region","cls","score","x","y","w_px","h_px","fold"]).to_parquet(f)
    print("saved",f,len(rows),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--fold",type=int); ap.add_argument("--oof",action="store_true")
    ap.add_argument("--tag",default="dsat"); ap.add_argument("--iters",type=int,default=2500); ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--init"); ap.add_argument("--tune-last",type=int,default=0)
    a=ap.parse_args()
    if a.fold is not None: train(a.fold,a.tag,a.iters,a.bs,init=a.init,tune_last=a.tune_last)
    if a.oof: oof(a.tag)
