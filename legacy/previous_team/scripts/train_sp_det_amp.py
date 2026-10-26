"""Optical expert, detector family: D-FINE or RT-DETRv2 (HF transformers, Apache-2.0; --ckpt picks) on prepared scenes.

    python scripts/train_sp_det.py --data work/sp --folds 0,1 [--ckpt ustc-community/dfine-large-coco] [--iters 8000]

Same data, folds and output schema as scripts/train_sp.py, so scripts/eval_sp.py scores it and the two families fuse:
* folds: 20 km groups (docs/site_groups.csv g20) balanced by object count into --k folds (identical algorithm).
* targets: label polygons -> boxes (cx, cy, w, h); chips of --chip px, 60% object-centred (class-balanced 1/sqrt(n),
  capped at 4x), rot90/flip, colour gain/offset, gamma, blur, grey; boxes less than half inside a chip are dropped.
* output polygons: the ellipse inscribed in each box. For compact objects it covers most of the label and its
  representative point lies inside it, which is what the official match needs (cov >= 0.5 and point-in-label).
* inference: tiles with overlap, per-class NMS across tiles, flip test-time augmentation (box average via WBF-lite).
Writes <outroot>/<tag>/fold<f>.pt, oof_pred<folds>.parquet (thr = 0), oof_gt.parquet, folds.json."""
import argparse, json, math, os, random, sys, time, warnings, zlib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, cv2
from shapely import wkb, affinity
from shapely.geometry import Point
from rasterio.transform import Affine
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="work/sp"); ap.add_argument("--folds", default="0,1,2,3"); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--ckpt", default="ustc-community/dfine-large-coco"); ap.add_argument("--iters", type=int, default=8000)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--chip", type=int, default=640); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--tag", default="sp_dfine_l"); ap.add_argument("--outroot", default="work/sp_runs"); ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min_score", type=float, default=0.05)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); AMP = DEV.type == "cuda" and os.environ.get("DET_AMP", "1") == "1"
if os.environ.get("NO_TF32") == "1":   # TF32 (Ada default for cuDNN convs) distorts D-FINE gradients by ~22 %
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
print("device", DEV, torch.cuda.get_device_name(0) if AMP else "", flush=True)
CLASSES = ["курганы_целые", "курганы_поврежденные", "городища", "фортификации", "архитектуры"]
OUT = f"{a.outroot}/{a.tag}"; os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ data (as scripts/train_sp.py)
SC = pd.read_csv(f"{a.data}/scenes.csv"); LB = pd.read_parquet(f"{a.data}/labels.parquet")
grp = pd.read_csv("docs/site_groups.csv").set_index("site")["g20"].to_dict()
SC = SC[SC.valid_frac > 0.01].reset_index(drop=True)
SCN = {}
for r in SC.itertuples():
    T = Affine(*json.loads(r.T)); L = LB[LB.scene == r.scene]; boxes = []
    for c, x in zip(L.cls, L.wkb):
        g = wkb.loads(x); x0, y0, x1, y1 = g.bounds
        c0, r0 = ~T * (x0, y1); c1, r1 = ~T * (x1, y0)
        boxes.append((CLASSES.index(c), min(c0, c1), min(r0, r1), max(c0, c1), max(r0, r1)))
    SCN[r.scene] = dict(site=r.site, T=T, img=np.load(f"{a.data}/img/{r.scene}.npy", mmap_mode="r"), boxes=np.array(boxes, np.float32).reshape(-1, 5),
                        g=grp.get(r.site, zlib.crc32(r.site.encode()) % 997), L=[(c, wkb.loads(x)) for c, x in zip(L.cls, L.wkb)])
gobj = {}
for s in SCN.values():
    gobj[s["g"]] = gobj.get(s["g"], 0) + len(s["boxes"]) + 1
fold_of_g, load = {}, [0] * a.k
for g, n in sorted(gobj.items(), key=lambda x: -x[1]):
    f = int(np.argmin(load)); fold_of_g[g] = f; load[f] += n
for s in SCN.values():
    s["fold"] = fold_of_g[s["g"]]
json.dump({k: v["fold"] for k, v in SCN.items()}, open(f"{OUT}/folds.json", "w", encoding="utf-8"), ensure_ascii=False)
SCALE = int(os.environ.get("DET_SCALE", "1"))     # coarse branch: scenes downsampled SCALE× (after the fold assignment,
if SCALE > 1:                                      # so folds stay identical); only boxes ≥ DET_MINBOX m kept as targets
    minbox_px = float(os.environ.get("DET_MINBOX", "40")) / (0.5 * SCALE)
    for s in SCN.values():
        a0 = np.asarray(s["img"]); C_, H_, W_ = a0.shape; h_, w_ = max(1, H_ // SCALE), max(1, W_ // SCALE)
        s["img"] = np.stack([cv2.resize(np.ascontiguousarray(a0[c_]), (w_, h_), interpolation=cv2.INTER_AREA) for c_ in range(C_)])
        s["T"] = s["T"] * Affine.scale(SCALE)
        B = s["boxes"].copy(); B[:, 1:] /= SCALE
        s["boxes"] = B[np.maximum(B[:, 3] - B[:, 1], B[:, 4] - B[:, 2]) >= minbox_px]
    print(f"coarse branch: ×{SCALE} ({0.5 * SCALE} m/px), boxes ≥ {minbox_px:.0f} px kept: {sum(len(s['boxes']) for s in SCN.values())}", flush=True)
cls_n = np.array([sum(int((s["boxes"][:, 0] == ci).sum()) for s in SCN.values()) for ci in range(len(CLASSES))], float)
print(f"scenes {len(SCN)} objects {int(cls_n.sum())} class counts {dict(zip(CLASSES, cls_n.astype(int)))} fold loads {load}", flush=True)


def _win(a0, n, L):
    """slice + (before, after) padding so that a window of n starting at a0 always has length n, even fully outside [0, L)."""
    s0, s1 = min(max(a0, 0), L), min(max(a0 + n, 0), L); before = max(0, min(-a0, n))
    return slice(s0, s1), (before, n - before - (s1 - s0))


def read_chip(s, y0, x0, n):
    S = SCN[s]; H, W = S["img"].shape[1:]
    (ys, py), (xs, px) = _win(y0, n, H), _win(x0, n, W); pad = (py, px)
    return np.pad(np.asarray(S["img"][:, ys, xs]), ((0, 0),) + pad)


def sample(train_scenes, n):
    pools = sample.pools.setdefault(tuple(train_scenes), {ci: [(s, i) for s in train_scenes for i, b in enumerate(SCN[s]["boxes"]) if b[0] == ci] for ci in range(len(CLASSES))})
    pools = {k: v for k, v in pools.items() if v}
    w = np.array([1 / math.sqrt(max(cls_n[k], 1)) for k in pools]); w = np.minimum(w, 4 * w.min()); w /= w.sum()
    if random.random() < 0.6 and pools:
        c = list(pools)[np.random.choice(len(pools), p=w)]
        if SITE_BAL:                     # site first, then an object: dense cemeteries no longer dominate
            by_site = sample.by_site.setdefault((tuple(train_scenes), c), {})
            if not by_site:
                for s_, i_ in pools[c]:
                    by_site.setdefault(SCN[s_]["site"], []).append((s_, i_))
            s, i = random.choice(by_site[random.choice(list(by_site))])
        else:
            s, i = random.choice(pools[c])
        b = SCN[s]["boxes"][i]
        cy, cx = (b[2] + b[4]) / 2 + random.uniform(-n / 3, n / 3), (b[1] + b[3]) / 2 + random.uniform(-n / 3, n / 3)
    else:
        s = random.choice(train_scenes); H, W = SCN[s]["img"].shape[1:]; cy, cx = random.uniform(0, H), random.uniform(0, W)
    y0, x0 = int(cy - n // 2), int(cx - n // 2); im = read_chip(s, y0, x0, n)
    B = SCN[s]["boxes"].copy(); B[:, [1, 3]] -= x0; B[:, [2, 4]] -= y0
    area = (B[:, 3] - B[:, 1]) * (B[:, 4] - B[:, 2]); Bc = B.copy(); Bc[:, 1:] = np.clip(Bc[:, 1:], 0, n)
    keep = ((Bc[:, 3] - Bc[:, 1]) * (Bc[:, 4] - Bc[:, 2]) >= 0.5 * np.maximum(area, 1e-6)) & (Bc[:, 3] - Bc[:, 1] >= 1) & (Bc[:, 4] - Bc[:, 2] >= 1)
    return augment(im, Bc[keep], n)
sample.pools = {}
sample.by_site = {}
SITE_BAL = os.environ.get("DET_SITE_BAL") == "1"
QAUG = os.environ.get("DET_QAUG") == "1"


def degrade(rgb):
    """Image-quality augmentation (low and high quality sources): resolution loss, JPEG artefacts, sensor noise, haze,
    or sharpening. rgb (3,n,n) float 0..255."""
    im = np.clip(rgb.transpose(1, 2, 0), 0, 255).astype(np.uint8); n = im.shape[0]
    if random.random() < 0.4:
        f = random.uniform(1.5, 3.2); m = max(8, int(n / f))
        im = cv2.resize(cv2.resize(im, (m, m), interpolation=cv2.INTER_AREA), (n, n), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if random.random() < 0.35:
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, random.randint(15, 60)]); im = cv2.imdecode(enc, 1)
    im = im.astype(np.float32)
    if random.random() < 0.3:
        im = im + np.random.normal(0, random.uniform(3, 12), im.shape).astype(np.float32)
    if random.random() < 0.2:
        a = random.uniform(0.1, 0.4); im = im * (1 - a) + a * random.uniform(150, 230)
    if random.random() < 0.15:
        bl = cv2.GaussianBlur(im, (0, 0), 1.2); im = im + random.uniform(0.5, 1.5) * (im - bl)
    return np.clip(im, 0, 255).transpose(2, 0, 1)


def augment(im, B, n):
    rgb = im[:3].astype(np.float32); vd = (im[3] > 0).astype(np.float32)
    if random.random() < 0.3:
        k = random.choice([3, 5, 7]); rgb = np.stack([cv2.GaussianBlur(c, (k, k), 0) for c in rgb])
    rgb = rgb * np.random.uniform(0.75, 1.3, (3, 1, 1)).astype(np.float32) * random.uniform(0.85, 1.15) + np.random.uniform(-20, 20, (3, 1, 1)).astype(np.float32)
    if random.random() < 0.3:
        rgb = 255 * np.clip(rgb / 255, 0, 1) ** random.uniform(0.7, 1.4)
    if random.random() < 0.15:
        rgb = np.repeat(rgb.mean(0, keepdims=True), 3, 0)
    if QAUG and random.random() < 0.5:
        rgb = degrade(rgb)
    rgb = np.clip(rgb, 0, 255) * vd
    for _ in range(random.randrange(4)):  # rot90 counter-clockwise: (x, y) -> (y, n - x)
        rgb = np.rot90(rgb, 1, axes=(1, 2)); B = B[:, [0, 2, 1, 4, 3]].copy(); B[:, 2], B[:, 4] = n - B[:, 4], n - B[:, 2]
    if random.random() < 0.5:
        rgb = rgb[:, :, ::-1]; B = B.copy(); B[:, 1], B[:, 3] = n - B[:, 3], n - B[:, 1]
    return np.ascontiguousarray(rgb) / 255.0, B


def to_target(B, n):
    cx, cy = (B[:, 1] + B[:, 3]) / 2 / n, (B[:, 2] + B[:, 4]) / 2 / n; w, h = (B[:, 3] - B[:, 1]) / n, (B[:, 4] - B[:, 2]) / n
    return {"class_labels": torch.as_tensor(B[:, 0], dtype=torch.long), "boxes": torch.as_tensor(np.stack([cx, cy, w, h], 1), dtype=torch.float32).reshape(-1, 4)}


MOSAIC = os.environ.get("DET_MOSAIC") == "1"


def sample_mosaic(train_scenes, n):
    """Mosaic (Global Wheat / YOLO): four n/2 crops — usually different scenes and sites — tiled into one n×n image."""
    h = n // 2; img = np.zeros((3, n, n), np.float32); Bs = []
    for k, (oy, ox) in enumerate(((0, 0), (0, h), (h, 0), (h, h))):
        im, B = sample(train_scenes, h)
        img[:, oy:oy + h, ox:ox + h] = im
        if len(B):
            B = B.copy(); B[:, [1, 3]] += ox; B[:, [2, 4]] += oy; Bs.append(B)
    return img, (np.concatenate(Bs) if Bs else np.zeros((0, 5), np.float32))


class _Batches(torch.utils.data.IterableDataset):
    def __init__(self, scenes, n): self.scenes, self.n = scenes, n
    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        if wi: random.seed(a.seed * 1000 + wi.id + int(time.time())); np.random.seed((a.seed * 1000 + wi.id + int(time.time())) % 2**32)
        for _ in range(self.n):
            X, Y = zip(*[(sample_mosaic(self.scenes, a.chip) if (MOSAIC and random.random() < 0.5) else sample(self.scenes, a.chip)) for _ in range(a.bs)])
            yield torch.from_numpy(np.stack(X).astype(np.float32)), [to_target(b, a.chip) for b in Y]


def make_model():
    from transformers import AutoModelForObjectDetection  # D-FINE or RT-DETRv2, chosen by --ckpt
    m = AutoModelForObjectDetection.from_pretrained(a.ckpt, num_labels=len(CLASSES), id2label=dict(enumerate(CLASSES)),
                                                label2id={c: i for i, c in enumerate(CLASSES)}, ignore_mismatched_sizes=True,
                                                **({"num_denoising": int(os.environ["DET_NUM_DN"])} if os.environ.get("DET_NUM_DN") else {}))
    return m.to(DEV)


def train(train_scenes, iters, tag):
    model = make_model(); model.train()
    bb = [p for n_, p in model.named_parameters() if "backbone" in n_]; oth = [p for n_, p in model.named_parameters() if "backbone" not in n_]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr * 0.1}, {"params": oth, "lr": a.lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.1, a.lr], total_steps=iters, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=AMP); t0 = time.time(); run = 0
    dl = torch.utils.data.DataLoader(_Batches(train_scenes, iters // max(a.workers, 1) + 2), batch_size=None, num_workers=a.workers,
                                     collate_fn=lambda b: b, prefetch_factor=4 if a.workers else None)
    src = iter(dl)
    for it in range(iters):
        x, ys = next(src)
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP):
            out = model(pixel_values=x.to(DEV, non_blocking=True), labels=[{k: v.to(DEV) for k, v in y.items()} for y in ys])
        loss = out.loss
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); scaler.step(opt); scaler.update(); sch.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if it % 50 == 0 or it == iters - 1:
            print(f"[{tag}] it {it}/{iters} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if it and it % 1000 == 0:
            torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
    torch.save(model.state_dict(), f"{OUT}/{tag}.pt"); model.eval()
    return model


@torch.no_grad()
def detect(model, x):
    """x: (B,3,n,n) in [0,1]. Returns per image (scores (Q,C) sigmoid, boxes (Q,4) cx,cy,w,h normalised), flip-TTA averaged."""
    with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP):
        o1 = model(pixel_values=x); o2 = model(pixel_values=torch.flip(x, [3]))
    s1, b1 = o1.logits.float().sigmoid(), o1.pred_boxes.float(); s2, b2 = o2.logits.float().sigmoid(), o2.pred_boxes.float()
    b2 = b2.clone(); b2[..., 0] = 1 - b2[..., 0]
    return [(torch.cat([s1[i], s2[i]]), torch.cat([b1[i], b2[i]])) for i in range(len(x))]


@torch.no_grad()
def predict_scene(model, s, n=None, ov=160):
    import torchvision
    n = n or a.chip; S = SCN[s]; H, W = S["img"].shape[1:]; step = n - ov
    tiles = [(y, x) for y in range(0, max(H - ov, 1), step) for x in range(0, max(W - ov, 1), step)]
    allb, alls, allc = [], [], []
    for k in range(0, len(tiles), 8):
        bt = tiles[k:k + 8]; X = []
        for y, x in bt:
            im = read_chip(s, y, x, n); X.append(im[:3].astype(np.float32) / 255.0 * (im[3] > 0))
        for (y, x), (sc, bx) in zip(bt, detect(model, torch.from_numpy(np.stack(X)).to(DEV))):
            score, cls = sc.max(1); m = score >= a.min_score
            if m.sum() > 100:  # at most 100 detections per tile (flip-TTA pair included)
                m &= score >= score[m].topk(100).values[-1]
            if not m.any():
                continue
            b = bx[m] * n; xy = torch.stack([b[:, 0] - b[:, 2] / 2 + x, b[:, 1] - b[:, 3] / 2 + y, b[:, 0] + b[:, 2] / 2 + x, b[:, 1] + b[:, 3] / 2 + y], 1)
            allb.append(xy); alls.append(score[m]); allc.append(cls[m])
    if not allb:
        return []
    B, Sx, C = torch.cat(allb), torch.cat(alls), torch.cat(allc)
    keep = torchvision.ops.batched_nms(B, Sx, C, 0.5)
    out = []
    for i in keep.tolist():
        x0, y0, x1, y1 = B[i].tolist()
        (X0, Y0), (X1, Y1) = S["T"] * (x0, y0), S["T"] * (x1, y1)
        e = affinity.scale(Point((X0 + X1) / 2, (Y0 + Y1) / 2).buffer(1.0, 16), abs(X1 - X0) / 2, abs(Y1 - Y0) / 2)
        if e.area > 0:
            out.append((CLASSES[int(C[i])], e, float(Sx[i])))
    return out


rows = []
folds = [int(x) for x in a.folds.split(",")]
for f in folds:
    tr = [s for s in SCN if SCN[s]["fold"] != f]; va = [s for s in SCN if SCN[s]["fold"] == f]
    if a.smoke:
        va = va[:2]; tr = tr or va
    if not va:
        continue
    model = train(tr, a.iters if not a.smoke else 10, f"fold{f}"); t0 = time.time()
    for s in va:
        for c, poly, sc in predict_scene(model, s):
            rows.append(dict(scene=s, site=SCN[s]["site"], fold=f, cls=c, thr=0.0, score=sc, wkb=poly.wkb))
    pd.DataFrame(rows, columns=["scene", "site", "fold", "cls", "thr", "score", "wkb"]).to_parquet(f"{OUT}/oof_pred{a.folds.replace(',', '')}.parquet")
    print(f"fold {f}: predicted {len(va)} scenes in {time.time() - t0:.0f}s, {len(rows)} candidates so far", flush=True)
gt = [dict(scene=s, site=S["site"], fold=S["fold"], cls=c, wkb=g.wkb) for s, S in SCN.items() for c, g in S["L"]]
pd.DataFrame(gt).to_parquet(f"{OUT}/oof_gt.parquet")
print("done")
