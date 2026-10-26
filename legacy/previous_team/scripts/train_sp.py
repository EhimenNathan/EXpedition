"""Satellite (SpOr) expert on CPU: U-Net (ImageNet encoder) -> 5 class masks + object-centre heatmap -> instances.

    python scripts/train_sp.py --folds 0,1,2 --iters 1500 [--enc resnet34] [--tag sp_r34] [--final]

Data: work/sp (scripts/prep_sp.py): 0.5 m RGB scenes (memmapped .npy) and labels in each scene's UTM CRS.
CV: sites grouped by 20 km clusters (docs/site_groups.csv, g20) into K folds balanced by object count; each fold's
scenes are predicted by a model that never saw their cluster. Out-of-fold candidates are saved for honest per-class
threshold selection under the official metric (scripts/eval_sp.py).
Sampling: 60% chips centred on objects (class-balanced, 1/sqrt(n), capped at 4x), 40% random valid chips.
Augmentation: rot90/flip, per-channel gain/offset, gamma, Gaussian blur (mimics coarse scenes upsampled to 0.5 m),
scale jitter 0.8-1.25. Loss: focal + Dice on classes (valid-masked) + CenterNet focal loss on the centre heatmap.
--final trains one model on all sites (for the submission)."""
import argparse, json, math, os, random, sys, time, warnings, zlib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import cv2
from shapely import wkb
from rasterio import features
from rasterio.transform import Affine
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import instances as INS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="work/sp"); ap.add_argument("--folds", default="0,1,2"); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--chip", type=int, default=256)
ap.add_argument("--enc", default="resnet18"); ap.add_argument("--dec", default="128,64,32,16,8"); ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--tag", default="sp_r34")
ap.add_argument("--final", action="store_true"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--outroot", default="work/sp_runs"); ap.add_argument("--maskdir", default=""); ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--masks_only", action="store_true"); ap.add_argument("--arch", default="Unet"); ap.add_argument("--save_prob", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); AMP = DEV.type == "cuda"
print("device", DEV, torch.cuda.get_device_name(0) if AMP else "", flush=True)
GSD = 0.5
CLASSES = ["курганы_целые", "курганы_поврежденные", "городища", "фортификации", "архитектуры"]
OUT = f"{a.outroot}/{a.tag}"; os.makedirs(OUT, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255; STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

# ------------------------------------------------------------------ data
SC = pd.read_csv(f"{a.data}/scenes.csv"); LB = pd.read_parquet(f"{a.data}/labels.parquet")
grp = pd.read_csv("docs/site_groups.csv").set_index("site")["g20"].to_dict()
SC = SC[SC.valid_frac > 0.01].reset_index(drop=True)
SCN = {}
for r in SC.itertuples():
    T = Affine(*json.loads(r.T)); img = np.load(f"{a.data}/img/{r.scene}.npy", mmap_mode="r")
    L = LB[LB.scene == r.scene]
    objs = []
    mpath = f"{a.maskdir or a.data + '/img'}/{r.scene}.mask.npy"; os.makedirs(os.path.dirname(mpath), exist_ok=True)
    if not os.path.exists(mpath):
        mask = np.zeros((r.H, r.W), np.uint8)
        for ci, c in enumerate(CLASSES):
            gs = [wkb.loads(x) for x in L[L.cls == c].wkb]
            if gs:
                mask |= features.rasterize([(g, 1) for g in gs], out_shape=mask.shape, transform=T, all_touched=True).astype(np.uint8) << ci
        np.save(mpath, mask)
    for c, x in zip(L.cls, L.wkb):
        g = wkb.loads(x); col, row = ~T * (g.centroid.x, g.centroid.y)
        objs.append((CLASSES.index(c), col, row, max(1.0, math.sqrt(g.area / math.pi) / GSD)))
    SCN[r.scene] = dict(site=r.site, T=T, img=img, mask=np.load(mpath, mmap_mode="r"), objs=objs, g=grp.get(r.site, zlib.crc32(r.site.encode()) % 997),
                        L=[(c, wkb.loads(x)) for c, x in zip(L.cls, L.wkb)], crs=r.crs)
print(f"scenes {len(SCN)}  objects {sum(len(s['objs']) for s in SCN.values())}  Mpx "
      f"{sum(s['img'].shape[1] * s['img'].shape[2] for s in SCN.values()) / 1e6:.0f}", flush=True)

if a.masks_only:
    sys.exit()
# folds: greedy balance of object counts over 20 km groups
gobj = {}
for s in SCN.values():
    gobj[s["g"]] = gobj.get(s["g"], 0) + len(s["objs"]) + 1
fold_of_g, load = {}, [0] * a.k
for g, n in sorted(gobj.items(), key=lambda x: -x[1]):
    f = int(np.argmin(load)); fold_of_g[g] = f; load[f] += n
for s in SCN.values():
    s["fold"] = fold_of_g[s["g"]]
json.dump({k: v["fold"] for k, v in SCN.items()}, open(f"{OUT}/folds.json", "w", encoding="utf-8"), ensure_ascii=False)
cls_n = np.array([sum(1 for s in SCN.values() for o in s["objs"] if o[0] == ci) for ci in range(len(CLASSES))], float)
print("class counts", dict(zip(CLASSES, cls_n.astype(int))), "fold loads", load, flush=True)


def _win(a0, n, L):
    """slice + (before, after) padding so that a window of n starting at a0 always has length n, even fully outside [0, L)."""
    s0, s1 = min(max(a0, 0), L), min(max(a0 + n, 0), L); before = max(0, min(-a0, n))
    return slice(s0, s1), (before, n - before - (s1 - s0))


def chip(s, cy, cx, n):
    S = SCN[s]; H, W = S["img"].shape[1:]
    y0, x0 = int(cy - n // 2), int(cx - n // 2)
    (ys, py), (xs, px) = _win(y0, n, H), _win(x0, n, W); pad = (py, px)
    im = np.pad(np.asarray(S["img"][:, ys, xs]), ((0, 0),) + pad)
    mb = np.pad(np.asarray(S["mask"][ys, xs]), pad)
    hm = np.zeros((n, n), np.float32)
    for ci, ox, oy, r in S["objs"]:
        u, v = ox - x0, oy - y0
        if -r < u < n + r and -r < v < n + r:
            sg = max(1.0, r / 3); xl, xh = int(max(u - 3 * sg, 0)), int(min(u + 3 * sg + 1, n)); yl, yh = int(max(v - 3 * sg, 0)), int(min(v + 3 * sg + 1, n))
            if xh > xl and yh > yl:
                yy, xx = np.mgrid[yl:yh, xl:xh].astype(np.float32)
                hm[yl:yh, xl:xh] = np.maximum(hm[yl:yh, xl:xh], np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sg * sg)))
    return im, mb, hm


def augment(im, mb, hm, n):
    rgb = im[:3].astype(np.float32); vd = (im[3] > 0).astype(np.float32)
    if random.random() < 0.5:  # scale jitter
        sc = random.uniform(0.8, 1.25); m2 = int(round(n * sc)); o = (m2 - n) // 2
        f = lambda x, interp: cv2.resize(x, (m2, m2), interpolation=interp)
        rgb = np.stack([f(c, cv2.INTER_LINEAR) for c in rgb]); vd = f(vd, cv2.INTER_NEAREST); mb = f(mb, cv2.INTER_NEAREST); hm = f(hm, cv2.INTER_LINEAR)
        if o >= 0:
            rgb, vd, mb, hm = rgb[:, o:o + n, o:o + n], vd[o:o + n, o:o + n], mb[o:o + n, o:o + n], hm[o:o + n, o:o + n]
        else:
            p = -o; q = n - m2 - p
            rgb = np.pad(rgb, ((0, 0), (p, q), (p, q))); vd = np.pad(vd, ((p, q), (p, q))); mb = np.pad(mb, ((p, q), (p, q))); hm = np.pad(hm, ((p, q), (p, q)))
    if random.random() < 0.3:
        k = random.choice([3, 5, 7]); rgb = np.stack([cv2.GaussianBlur(c, (k, k), 0) for c in rgb])
    gain = np.random.uniform(0.75, 1.3, (3, 1, 1)).astype(np.float32) * random.uniform(0.85, 1.15)
    rgb = rgb * gain + np.random.uniform(-20, 20, (3, 1, 1)).astype(np.float32)
    if random.random() < 0.3:
        rgb = 255 * np.clip(rgb / 255, 0, 1) ** random.uniform(0.7, 1.4)
    if random.random() < 0.15:
        rgb = np.repeat(rgb.mean(0, keepdims=True), 3, 0)  # grey scenes exist
    rgb = np.clip(rgb, 0, 255) * vd
    k = random.randrange(4); fl = random.random() < 0.5
    g = lambda x: np.ascontiguousarray(np.flip(np.rot90(x, k, axes=(-2, -1)), -1) if fl else np.rot90(x, k, axes=(-2, -1)))
    return g(rgb), g(mb), g(hm), g(vd)


POOLS = {}


def batch(train_scenes, bs, n):
    key = tuple(train_scenes)
    if key not in POOLS:
        POOLS[key] = {ci: [(s, i) for s in train_scenes for i, o in enumerate(SCN[s]["objs"]) if o[0] == ci] for ci in range(len(CLASSES))}
        POOLS[key] = {k: v for k, v in POOLS[key].items() if v}
    pools = POOLS[key]
    w = np.array([1 / math.sqrt(max(cls_n[k], 1)) for k in pools]); w = np.minimum(w, 4 * w.min()); w /= w.sum()
    X, Y, V, Hm = [], [], [], []
    for _ in range(bs):
        if random.random() < 0.6 and pools:
            c = list(pools)[np.random.choice(len(pools), p=w)]; s, i = random.choice(pools[c]); _, ox, oy, _ = SCN[s]["objs"][i]
            cy, cx = oy + random.uniform(-n / 3, n / 3), ox + random.uniform(-n / 3, n / 3)
        else:
            s = random.choice(train_scenes); H, W = SCN[s]["img"].shape[1:]
            cy, cx = random.uniform(0, H), random.uniform(0, W)
        im, mb, hm = chip(s, cy, cx, n)
        rgb, mb, hm, vd = augment(im, mb, hm, n)
        X.append((rgb - MEAN[:, None, None]) / STD[:, None, None] * vd); Y.append(np.stack([(mb >> ci) & 1 for ci in range(len(CLASSES))])); V.append(vd); Hm.append(hm)
    t = lambda x: torch.from_numpy(np.stack(x).astype(np.float32))
    return t(X), t(Y), t(V)[:, None], t(Hm)[:, None]


def dice_loss(p, y, v):
    inter = (p * y * v).sum((2, 3)); den = ((p + y) * v).sum((2, 3)); has = (y * v).sum((2, 3)) > 0
    return ((1 - (2 * inter + 1) / (den + 1)) * has).sum() / has.sum().clamp(min=1)


def loss_fn(out, y, v, hm):
    lg = out[:, :len(CLASSES)]; p = torch.sigmoid(lg)
    focal = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * ((1 - p) * y + p * (1 - y)) ** 2 * v).mean() * 10
    ph = torch.sigmoid(out[:, -1:]).clamp(1e-4, 1 - 1e-4); pos = (hm > 0.99).float()
    hl = (-(pos * (1 - ph) ** 2 * torch.log(ph) + (1 - pos) * (1 - hm) ** 4 * ph ** 2 * torch.log(1 - ph)) * v).sum() / pos.sum().clamp(min=1)
    return focal + dice_loss(p, y, v) + hl


def make_model():
    import segmentation_models_pytorch as smp
    kw = dict(decoder_channels=tuple(int(x) for x in a.dec.split(","))) if a.arch in ("Unet", "UnetPlusPlus") else {}
    m = getattr(smp, a.arch)(a.enc, encoder_weights="imagenet", in_channels=3, classes=len(CLASSES) + 1, **kw)  # half-width decoder: 2.9 s/it vs 7.2 s (r34, default) on this CPU
    head = [x for x in m.segmentation_head.modules() if isinstance(x, nn.Conv2d)][-1]
    with torch.no_grad():
        head.bias[:len(CLASSES)].fill_(-2.2); head.bias[len(CLASSES):].fill_(-4.6)
    return m.to(DEV)


@torch.no_grad()
def predict_scene(model, s, n=512, ov=64):
    S = SCN[s]; H, W = S["img"].shape[1:]; step = n - ov
    acc = np.zeros((len(CLASSES) + 1, H, W), np.float32); ws = np.zeros((H, W), np.float32)
    w1 = np.hanning(n + 2)[1:-1]; win = (np.outer(w1, w1) + 1e-3).astype(np.float32)
    tiles = [(y, x) for y in range(0, max(H - ov, 1), step) for x in range(0, max(W - ov, 1), step)]
    for k in range(0, len(tiles), 8):
        bt = tiles[k:k + 8]; X = []
        for y, x in bt:
            im, _, _ = chip(s, y + n // 2, x + n // 2, n)
            vd = (im[3] > 0).astype(np.float32); X.append((im[:3].astype(np.float32) - MEAN[:, None, None]) / STD[:, None, None] * vd)
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP):
            xb = torch.from_numpy(np.stack(X)).to(DEV); o = model(xb).float()
            o = (o + torch.flip(model(torch.flip(xb, [3])).float(), [3])) / 2  # flip test-time augmentation
        pr = torch.sigmoid(o).cpu().numpy()
        for (y, x), p in zip(bt, pr):
            hh, ww = min(n, H - y), min(n, W - x)
            acc[:, y:y + hh, x:x + ww] += p[:, :hh, :ww] * win[:hh, :ww]; ws[y:y + hh, x:x + ww] += win[:hh, :ww]
    acc /= np.maximum(ws, 1e-6); acc[:, np.asarray(S["img"][3]) == 0] = 0
    return acc


class _Batches(torch.utils.data.IterableDataset):
    def __init__(self, scenes, n_batches): self.scenes, self.n = scenes, n_batches
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); k = wi.num_workers if wi else 1
        if wi: random.seed(a.seed * 1000 + wi.id + int(time.time())); np.random.seed((a.seed * 1000 + wi.id + int(time.time())) % 2**32)
        for _ in range(self.n // k + 1): yield batch(self.scenes, a.bs, a.chip)


def batches(scenes, iters):
    if a.workers <= 0:
        while True: yield batch(scenes, a.bs, a.chip)
    dl = torch.utils.data.DataLoader(_Batches(scenes, iters + 50), batch_size=None, num_workers=a.workers, pin_memory=AMP, prefetch_factor=4, persistent_workers=False)
    yield from dl


def train(train_scenes, iters, tag):
    model = make_model(); model.train()
    enc = [p for n_, p in model.named_parameters() if n_.startswith("encoder")]; oth = [p for n_, p in model.named_parameters() if not n_.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * 0.5}, {"params": oth, "lr": a.lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.5, a.lr], total_steps=iters, pct_start=0.1)
    t0 = time.time(); run = 0; scaler = torch.amp.GradScaler(enabled=AMP); src = batches(train_scenes, iters)
    for it in range(iters):
        x, y, v, hm = (t.to(DEV, non_blocking=True) for t in next(src))
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP):
            out = model(x)
        loss = loss_fn(out.float(), y, v, hm)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if it % 50 == 0 or it == iters - 1:
            print(f"[{tag}] it {it}/{iters} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if it and it % 500 == 0:
            torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
    torch.save(model.state_dict(), f"{OUT}/{tag}.pt"); model.eval()
    return model


if a.final:
    train(list(SCN), a.iters if not a.smoke else 20, "final")
    sys.exit()
rows = []
for f in [int(x) for x in a.folds.split(",")]:
    tr = [s for s in SCN if SCN[s]["fold"] != f]; va = [s for s in SCN if SCN[s]["fold"] == f]
    if a.smoke:
        va = va[:2]
    model = train(tr, a.iters if not a.smoke else 20, f"fold{f}")
    t0 = time.time()
    for s in va:
        acc = predict_scene(model, s)
        if a.save_prob: np.save(f"{OUT}/prob_{s}.npy", (acc * 255).astype(np.uint8))
        for thr in (0.2, 0.3, 0.4, 0.5):
            for c, poly, sc in INS.extract(acc[:len(CLASSES)], acc[-1], SCN[s]["T"], CLASSES, thr=thr, gsd=GSD):
                rows.append(dict(scene=s, site=SCN[s]["site"], fold=f, cls=c, thr=thr, score=sc, wkb=poly.wkb))
    pd.DataFrame(rows).to_parquet(f"{OUT}/oof_pred{a.folds.replace(',', '')}.parquet")
    print(f"fold {f}: predicted {len(va)} scenes in {time.time() - t0:.0f}s, {len(rows)} candidates so far", flush=True)
gt = [dict(scene=s, site=S["site"], fold=S["fold"], cls=c, wkb=g.wkb) for s, S in SCN.items() for c, g in S["L"]]
pd.DataFrame(gt).to_parquet(f"{OUT}/oof_gt.parquet")
print("done")
