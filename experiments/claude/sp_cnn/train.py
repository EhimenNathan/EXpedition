"""Satellite kurgan / architecture / fortification detector trained on CPU: ImageNet ResNet-18 (timm) + FPN decoder →
CenterNet heatmaps at stride 4 (2 m at 0.5 m GSD). Out of fold over koz3.sp_prep.site_folds (20 km clusters).

Channels: 0 kurgan centre heat, 1 damaged share (masked), 2 architecture heat, 3 fortification heat.
Crops 256 px (128 m): 60 % centred near a labelled object (class-balanced), 40 % random; augmentation: d4 (flips +
rot90), scale ×0.8–1.25, brightness / contrast / gamma, colour jitter, Gaussian blur, noise.
  python experiments/claude/sp_cnn/train.py [iters=3000] [folds=0,1,2,3,-1] [arch=resnet18|resnet50|satlas_swinb]
Device: CNN_DEVICE (default cuda when available); CNN_LR, CNN_BS override the learning rate / batch size;
SP_CNN_TAG names the OOF folder (default = arch).
OOF maps → work/sp_cnn/oof/<idx>.npz (p (4, H/4, W/4) float16, T of the map grid, crs, valid, region, tif).
"""
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
W_ = os.path.join(ROOT, os.environ.get("CNN_WORK", "work/sp_cnn"))
MOD = os.environ.get("KOZ3_MOD", "SpOr")
DEV = os.environ.get("CNN_DEVICE") or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
CROP, STRIDE, NCH = 256, 4, 4
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255; STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


class SatlasSwin:
    """SatlasPretrain aerial Swin-V2-B (allenai/satlas-pretrain aerial_swinb_si.pth, 0.5–2 m NAIP imagery) as a
    4-level feature extractor (strides 4 / 8 / 16 / 32). Satlas expects RGB in 0–1, so ImageNet-normalised input is
    converted back inside the module."""

    @staticmethod
    def make(pretrained=True):
        import torch
        import torch.nn as nn
        import torchvision

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = torchvision.models.swin_v2_b().features
                if pretrained:
                    from huggingface_hub import hf_hub_download
                    sd = torch.load(hf_hub_download("allenai/satlas-pretrain", "aerial_swinb_si.pth"), map_location="cpu")
                    self.body.load_state_dict({k[len("backbone.backbone.features."):]: v for k, v in sd.items()
                                               if k.startswith("backbone.backbone.features.")})
                self.register_buffer("m", torch.tensor(MEAN / 255).view(1, 3, 1, 1).float())
                self.register_buffer("s", torch.tensor(STD / 255).view(1, 3, 1, 1).float())

            def forward(self, x):
                x = (x * self.s + self.m).clamp(0, 1); out = []
                for i, layer in enumerate(self.body):
                    x = layer(x)
                    if i in (1, 3, 5, 7):
                        out.append(x.permute(0, 3, 1, 2))
                return out

            def channels(self):
                return [128, 256, 512, 1024]
        return Enc()


def build(arch="resnet18", pretrained=True):
    import timm
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "satlas_swinb":
                self.enc = SatlasSwin.make(pretrained); ch = self.enc.channels()
            else:
                self.enc = timm.create_model(arch + (".a1_in1k" if pretrained else ""), pretrained=pretrained, features_only=True,
                                             out_indices=(1, 2, 3, 4))
                ch = self.enc.feature_info.channels()
            self.lat = nn.ModuleList([nn.Conv2d(c, 96, 1) for c in ch])
            self.out = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(), nn.Conv2d(96, NCH, 1))
            nn.init.constant_(self.out[-1].bias, 0.0); self.out[-1].bias.data[[0, 2, 3]] = -4.6

        def forward(self, x):
            fs = self.enc(x)
            y = self.lat[-1](fs[-1])
            for f, l in zip(fs[-2::-1], list(self.lat)[-2::-1]):
                y = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False) + l(f)
            return self.out(y)                                     # stride 4
    return Net()


def load_all():
    S = []
    for m in sorted(glob.glob(os.path.join(W_, "meta", "*.npz"))):
        z = np.load(m, allow_pickle=True); k = os.path.basename(m)[:-4]
        S.append(dict(idx=k, img=np.load(os.path.join(W_, "img", k + ".npy"), mmap_mode="r"), objs=z["objs"], T=z["T"],
                      crs=str(z["crs"]), valid=z["valid"], region=str(z["region"]), tif=str(z["tif"])))
    return S


def heat_targets(objs, r0, c0, size, scale):
    """Targets for a crop starting at (r0, c0) of `size` source px, resampled by `scale` → (NCH, CROP/4, CROP/4), mask."""
    g = CROP // STRIDE
    Y = np.zeros((NCH, g, g), np.float32); M = np.zeros((g, g), bool)
    yy, xx = np.mgrid[0:g, 0:g]
    for row, col, sig, cid, dmg in objs:
        r = (row - r0) * scale / STRIDE - 0.5; c = (col - c0) * scale / STRIDE - 0.5; s = max(sig * scale / STRIDE, 1.0)
        if not (-3 * s <= r < g + 3 * s and -3 * s <= c < g + 3 * s):
            continue
        ch = (0, 2, 3)[int(cid)]
        G = np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * s * s))
        Y[ch] = np.maximum(Y[ch], G)
        ir, ic = int(round(r)), int(round(c))
        if 0 <= ir < g and 0 <= ic < g:
            Y[ch, ir, ic] = 1.0
        if cid == 0:
            near = G > 0.5; Y[1][near] = dmg; M |= near
    return Y, M


def sample(S, rng, batch):
    import cv2
    X, Y, M, V = [], [], [], []
    pool = [(i, j) for i, s in enumerate(S) for j in range(len(s["objs"])) if 0 <= s["objs"][j][0] < s["valid"][0] and 0 <= s["objs"][j][1] < s["valid"][1]]
    cls = np.array([S[i]["objs"][j][3] for i, j in pool]); cnt = np.bincount(cls.astype(int), minlength=3)
    pw = 1 / np.sqrt(cnt[cls.astype(int)]); pw /= pw.sum()
    sw = np.array([np.sqrt(s["valid"][0] * s["valid"][1]) for s in S]); sw /= sw.sum()
    while len(X) < batch:
        scale = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)))); size = int(round(CROP / scale))
        if rng.random() < 0.6 and pool:
            i, j = pool[rng.choice(len(pool), p=pw)]; s = S[i]
            cr, cc = s["objs"][j][:2]
            r0 = int(cr - rng.uniform(0.15, 0.85) * size); c0 = int(cc - rng.uniform(0.15, 0.85) * size)
        else:
            s = S[rng.choice(len(S), p=sw)]
            r0 = int(rng.integers(-size // 4, max(int(s["valid"][0]) - size * 3 // 4, 1)))
            c0 = int(rng.integers(-size // 4, max(int(s["valid"][1]) - size * 3 // 4, 1)))
        H, W_img = s["img"].shape[:2]
        a = np.full((size, size, 3), 0, np.uint8); v = np.zeros((size, size), np.uint8)
        ra, ca, rb, cb = max(r0, 0), max(c0, 0), min(r0 + size, int(s["valid"][0])), min(c0 + size, int(s["valid"][1]))
        if rb <= ra or cb <= ca:
            continue
        a[ra - r0:rb - r0, ca - c0:cb - c0] = s["img"][ra:rb, ca:cb]; v[ra - r0:rb - r0, ca - c0:cb - c0] = 1
        a = cv2.resize(a, (CROP, CROP), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        v = cv2.resize(v, (CROP // STRIDE, CROP // STRIDE), interpolation=cv2.INTER_NEAREST).astype(bool)
        y, m = heat_targets(s["objs"], r0, c0, size, CROP / size)
        x = a.astype(np.float32)                                              # photometric augmentation
        x = (x - x.mean()) * rng.uniform(0.7, 1.3) + x.mean() + rng.uniform(-20, 20)
        x = x * rng.uniform(0.9, 1.1, 3)[None, None]
        x = 255 * (np.clip(x, 0, 255) / 255) ** rng.uniform(0.8, 1.25)
        if rng.random() < 0.3:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
        if rng.random() < 0.3:
            x = x + rng.normal(0, rng.uniform(2, 8), x.shape)
        k = int(rng.integers(8))                                              # d4
        x, y, m, v = (np.rot90(t, k % 4, axes=(-2, -1) if t.ndim == 3 and t.shape[0] == NCH else (0, 1)) for t in (x, y, m, v))
        if k >= 4:
            x, y, m, v = x[:, ::-1], y[:, :, ::-1], m[:, ::-1], v[:, ::-1]
        X.append(((np.clip(x, 0, 255) - MEAN) / STD).transpose(2, 0, 1)); Y.append(y); M.append(m); V.append(v)
    return [np.ascontiguousarray(np.stack(t)) for t in (X, Y, M, V)]


def focal(pred, gt, valid):
    import torch
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = (gt >= 0.999).float() * valid; neg = (1 - pos) * valid
    return (-(torch.log(p) * (1 - p) ** 2 * pos).sum() - (torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg).sum()) / pos.sum().clamp(min=1)


def loss_fn(out, Y, M, V):
    import torch.nn.functional as F
    l = sum(focal(out[:, ch], Y[:, ch], V) for ch in (0, 2, 3))
    if M.any():
        l = l + F.binary_cross_entropy_with_logits(out[:, 1][M], Y[:, 1][M])
    return l


def predict_scene(model, img, valid, tile=512, step=384, dev=None):
    """Sliding window + horizontal-flip TTA → (NCH, ceil(H/4), ceil(W/4)) probabilities over the valid image."""
    import torch
    H, W = int(valid[0]), int(valid[1])
    gh, gw = -(-H // STRIDE), -(-W // STRIDE)
    acc = np.zeros((NCH, gh, gw), np.float32); cnt = np.zeros((gh, gw), np.float32)
    rs = list(range(0, max(H - tile, 0) + 1, step)); rs += [max(H - tile, 0)] if rs[-1] != max(H - tile, 0) else []
    cs = list(range(0, max(W - tile, 0) + 1, step)); cs += [max(W - tile, 0)] if cs[-1] != max(W - tile, 0) else []
    wy = np.minimum(np.arange(tile // STRIDE) + 1, tile // STRIDE - np.arange(tile // STRIDE)).astype(np.float32)
    wgt = np.minimum.outer(wy, wy)
    for r in rs:
        for c in cs:
            a = np.zeros((tile, tile, 3), np.float32); part = np.asarray(img[r:r + tile, c:c + tile], np.float32)
            a[:part.shape[0], :part.shape[1]] = part
            dev = dev or DEV
            x = torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1)[None].copy()).to(dev)
            with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
                o = torch.sigmoid(model(torch.cat([x, torch.flip(x, (3,))])).float())
                o = ((o[0] + torch.flip(o[1], (2,))) / 2).cpu().numpy()
            gr, gc = r // STRIDE, c // STRIDE
            h, w = min(tile // STRIDE, gh - gr), min(tile // STRIDE, gw - gc)
            acc[:, gr:gr + h, gc:gc + w] += o[:, :h, :w] * wgt[:h, :w]; cnt[gr:gr + h, gc:gc + w] += wgt[:h, :w]
    return acc / np.maximum(cnt, 1e-6)


def main(iters=3000, folds="0,1,2,3,-1", arch="resnet18", seed=0):
    import geopandas as gpd
    import torch
    from rasterio.transform import Affine
    from koz3.sp_prep import site_folds
    torch.set_num_threads(os.cpu_count()); torch.manual_seed(seed)
    print(f"device {DEV}", flush=True)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); fo = site_folds(L[L["mod"] == MOD])
    S = load_all(); fold = np.array([fo[s["region"]] for s in S])
    print(f"{len(S)} scenes, objects {int(sum(len(s['objs']) for s in S))}", flush=True)
    tag = os.environ.get("SP_CNN_TAG", arch)
    os.makedirs(os.path.join(W_, "oof_" + tag), exist_ok=True); os.makedirs(os.path.join(W_, "models"), exist_ok=True)
    for f in [int(x) for x in str(folds).split(",")]:
        tr = [S[i] for i in range(len(S)) if fold[i] != f]; rng = np.random.default_rng(seed * 100 + f + 1)
        lr = float(os.environ.get("CNN_LR", 2e-4 if arch == "satlas_swinb" else 5e-4))
        bs = int(os.environ.get("CNN_BS", 12))
        model = build(arch).to(DEV); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=int(iters), pct_start=0.1)
        t0 = time.time(); model.train()
        for it in range(int(iters)):
            X, Y, M, V = (torch.from_numpy(a).to(DEV) for a in sample(tr, rng, bs))
            with torch.autocast(DEV, dtype=torch.bfloat16):                # bf16 (CPU AMX ~2.7× faster; GPU autocast)
                out = model(X.float())
            loss = loss_fn(out.float(), Y, M, V.float())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % 200 == 0:
                print(f"  fold {f} it {it} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        model.eval(); torch.save(model.state_dict(), os.path.join(W_, "models", f"{tag}_f{f}_s{seed}.pt"))
        if f >= 0:
            for i in np.flatnonzero(fold == f):
                s = S[i]; p = predict_scene(model, s["img"], s["valid"])
                T = Affine(*s["T"]) * Affine.scale(STRIDE, STRIDE)
                np.savez_compressed(os.path.join(W_, "oof_" + tag, s["idx"] + ".npz"), p=p.astype(np.float16), T=np.array(T[:6]),
                                    crs=s["crs"], valid=np.array(p.shape[1:]), region=s["region"], tif=s["tif"])
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
