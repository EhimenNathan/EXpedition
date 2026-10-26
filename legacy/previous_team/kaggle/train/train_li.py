"""KOZ-3 LiDAR expert: train on DTM relief channels, evaluate out-of-fold with the competition-style object metric.
Env (all optional): ARCH=Unet ENC=resnet34 EPOCHS=20 ITERS=250 BS=16 CHIP=384 FOLDS=0,1,2,3,4 NFOLD=5 LR=3e-4
                    LOSS=bce|bce_dice|focal_dice|tversky  ENC_KW={json}  PRED=512  CH=all|hs|relief  HEAT=1  TAG=name  SMOKE=0"""
import os, sys, glob, json, math, time, random, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"], check=False)
import numpy as np, pandas as pd, geopandas as gpd, rasterio, torch, torch.nn as nn, torch.nn.functional as F
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage as ndi
import segmentation_models_pytorch as smp

DATA = os.environ.get("KOZ3_DATA", "/kaggle/input")
META = os.environ.get("KOZ3_META") or os.path.dirname(os.path.dirname(glob.glob(f"{DATA}/**/koz3/dtm.py", recursive=True)[0]))
sys.path.insert(0, META)
from koz3 import relief_torch as RT, metric as MET, instances as INS

E = lambda k, d: type(d)(os.environ.get(k, d))
ARCH, ENC, EPOCHS, ITERS, BS, CHIP = E("ARCH", "Unet"), E("ENC", "resnet34"), E("EPOCHS", 20), E("ITERS", 250), E("BS", 16), E("CHIP", 384)
NFOLD, LR, LOSS, CH, HEAT, TAG, SMOKE = E("NFOLD", 5), E("LR", 3e-4), E("LOSS", "focal_dice"), E("CH", "all"), E("HEAT", 1), E("TAG", "li_v1"), E("SMOKE", 0)
FOLDS = [int(f) for f in E("FOLDS", "0,1,2,3,4").split(",")]
GSD = 0.25; PADPX = int(RT.PAD_M / GSD); CROP_M = float(os.environ.get("CROP_M", 300))
CLASSES = ["курганы_целые", "курганы_поврежденные", "городища", "фортификации", "архитектуры"]
OUT = os.path.join(os.environ.get("KOZ3_WORK", "/kaggle/working"), TAG); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cpu": torch.set_num_threads(os.cpu_count() or 4)
torch.manual_seed(0); np.random.seed(0); random.seed(0)
print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

# ------------------------------------------------------------------ data
reg = {}
for f in glob.glob(f"{DATA}/**/register_sites.csv", recursive=True):
    R = pd.read_csv(f); reg = {r.site: (r.dx_m, r.dy_m) for r in R.itertuples() if isinstance(getattr(r, "dx_m", None), float) and r.ncc_after > r.ncc_before + 0.02 and r.ncc_after > 0.6}
print("label shifts applied:", {k: v for k, v in reg.items() if math.hypot(*v) > 0.75}, flush=True)
groups = pd.read_csv(f"{META}/docs/site_groups.csv").set_index("site")["g20"].to_dict()
SITES = {}
_dtms = {}
for p in sorted(glob.glob(f"{DATA}/**/lidar*/*/dtm.tif", recursive=True)):
    s_ = os.path.basename(os.path.dirname(p))
    if s_ not in _dtms or "prep-ground" in p or "lidar_ground" in p: _dtms[s_] = p  # ground-filtered DTM wins for unclassified LAS sites
for site, p in sorted(_dtms.items()):
    utm = json.load(open(f"{META}/data_meta/{site}/UTM.json", encoding="utf-8"))["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")
    L = []
    for f in glob.glob(f"{META}/data_meta/{site}/**/*.geojson", recursive=True):
        b = os.path.basename(f).replace("Ае", "Ae")
        if "_Li_" not in b: continue
        c = b.split("_Li_")[-1][:-8]
        if c not in CLASSES: continue
        g = gpd.read_file(f)
        if g.crs is None or not len(g): continue
        g = g.to_crs(utm); g = g[g.geometry.notna() & ~g.geometry.is_empty]; g["cls"] = c; L.append(g[["cls", "geometry"]])
    if not L: continue
    L = gpd.GeoDataFrame(pd.concat(L, ignore_index=True), crs=utm)
    dx, dy = reg.get(site, (0.0, 0.0))
    if dx or dy: L["geometry"] = L.geometry.translate(-dx, -dy)  # move labels from the _g frame into the DTM frame
    from rasterio.windows import from_bounds as _fb
    with rasterio.open(p) as r:
        bx = L.total_bounds; B = CROP_M
        win = _fb(bx[0] - B, bx[1] - B, bx[2] + B, bx[3] + B, r.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, r.width, r.height))
        dtm = r.read(1, window=win); T = r.window_transform(win)
    valid = dtm > -9000
    if valid.mean() < 0.05: continue
    dtm = np.where(valid, dtm, np.median(dtm[valid])).astype(np.float32)
    ip = p.replace("dtm.tif", "inten.tif")
    inten = rasterio.open(ip).read(1, window=win) if os.path.exists(ip) else np.zeros_like(dtm, np.uint8)
    mask = np.zeros(dtm.shape, np.uint8)  # bit ci set = class ci present
    for ci, c in enumerate(CLASSES):
        gg = L[L.cls == c]
        if len(gg): mask |= (features.rasterize([(g, 1) for g in gg.geometry], out_shape=dtm.shape, transform=T, all_touched=True).astype(np.uint8) << ci)
    cen = [(~T * (g.centroid.x, g.centroid.y), max(1.0, math.sqrt(g.area / math.pi) / GSD)) for g in L.geometry]
    cxy = np.array([[c[0][0], c[0][1], c[1]] for c in cen], np.float32).reshape(-1, 3)
    SITES[site] = dict(cxy=cxy, dtm=dtm, inten=inten, valid=valid, mask=mask, T=T, L=L, cen=cen, crs=utm, g=groups.get(site, -1))
print("sites:", len(SITES), "objects:", sum(len(s["L"]) for s in SITES.values()), "Mpx:", round(sum(s["valid"].size for s in SITES.values()) / 1e6, 1), flush=True)
if SMOKE: SITES = dict(list(SITES.items())[:6])

QS = 25.0  # int8 quantisation scale of the relief channels (values are clamped to +-5)


@torch.no_grad()
def precompute_relief(S, tile=1536):
    H, W = S["dtm"].shape; P = PADPX; out = np.zeros((len(RT.CHANNELS), H, W), np.int8)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y0, x0, y1, x1 = max(y - P, 0), max(x - P, 0), min(y + tile + P, H), min(x + tile + P, W)
            d = torch.from_numpy(np.ascontiguousarray(S["dtm"][y0:y1, x0:x1])).to(dev)[None, None]
            r = RT.relief(d, GSD)[0]
            hh, ww = min(tile, H - y), min(tile, W - x)
            r = r[:, y - y0:y - y0 + hh, x - x0:x - x0 + ww]
            out[:, y:y + hh, x:x + ww] = (r * QS).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
    S["rel"] = out; del S["dtm"]


_t = time.time()
RC = os.environ.get("RELIEF_CACHE")  # shared on-disk cache: processes memory-map the same pages
for _name, _s in SITES.items():
    fn = os.path.join(RC, _name + ".npy") if RC else None
    if fn and os.path.exists(fn):
        _s["rel"] = np.load(fn, mmap_mode="r"); del _s["dtm"]
    else:
        precompute_relief(_s)
        if fn:
            os.makedirs(RC, exist_ok=True); np.save(fn + ".tmp.npy", _s["rel"]); os.replace(fn + ".tmp.npy", fn)
            _s["rel"] = np.load(fn, mmap_mode="r")
print("relief ready in %.0fs, %.1f GB int8" % (time.time() - _t, sum(v["rel"].nbytes for v in SITES.values()) / 1e9), flush=True)
if os.environ.get("PRECOMPUTE_ONLY"): print("PRECOMPUTE_ONLY done"); sys.exit(0)

# grouped folds over 20-km clusters, balanced by object count
gl = pd.DataFrame([(s, d["g"], len(d["L"])) for s, d in SITES.items()], columns=["site", "g", "n"])
gsum = gl.groupby("g").n.sum().sort_values(ascending=False)
fold_of_g, load = {}, [0] * NFOLD
for g, n in gsum.items():
    k = int(np.argmin(load)); fold_of_g[g] = k; load[k] += n
gl["fold"] = gl.g.map(fold_of_g); gl.to_csv(f"{OUT}/folds.csv", index=False)
print(gl.groupby("fold").agg(sites=("site", "size"), objs=("n", "sum")).to_string(), flush=True)

IN_CH = {"all": 13, "relief": 12, "hs": 2}[CH]


def chip(site, cy, cx, n):
    S = SITES[site]; H, W = S["valid"].shape
    y0, x0 = int(cy - n // 2), int(cx - n // 2)
    ys, xs = slice(max(y0, 0), min(y0 + n, H)), slice(max(x0, 0), min(x0 + n, W))
    pad = ((max(-y0, 0), max(y0 + n - H, 0)), (max(-x0, 0), max(x0 + n - W, 0)))
    rel = np.pad(S["rel"][:, ys, xs], ((0, 0),) + pad)
    it = np.pad(S["inten"][ys, xs], pad)
    mb = np.pad(S["mask"][ys, xs], pad)
    mk = np.stack([(mb >> ci) & 1 for ci in range(len(CLASSES))]).astype(np.uint8)
    vd = np.pad(S["valid"][ys, xs], pad)
    hm = np.zeros((n, n), np.float32); C = S["cxy"]
    if len(C):
        u = C[:, 0] - x0; v = C[:, 1] - y0; r = C[:, 2]
        sel = (u > -r) & (u < n + r) & (v > -r) & (v < n + r)
        if sel.any():
            yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
            for uu, vv, rr in zip(u[sel], v[sel], r[sel]):
                sg = max(1.0, rr / 3); x_lo, x_hi = int(max(uu - 3 * sg, 0)), int(min(uu + 3 * sg + 1, n)); y_lo, y_hi = int(max(vv - 3 * sg, 0)), int(min(vv + 3 * sg + 1, n))
                if x_hi <= x_lo or y_hi <= y_lo: continue
                g = np.exp(-((xx[y_lo:y_hi, x_lo:x_hi] - uu) ** 2 + (yy[y_lo:y_hi, x_lo:x_hi] - vv) ** 2) / (2 * sg * sg))
                hm[y_lo:y_hi, x_lo:x_hi] = np.maximum(hm[y_lo:y_hi, x_lo:x_hi], g)
    return rel, it, mk, vd, hm


def to_input(d, it):
    """d: (B,1,12,n,n) int8 relief (as float or int8 tensor); it: (B,1,n,n) intensity uint8."""
    r = d[:, 0].float() / QS
    if CH == "hs": r = r[:, 7:9]
    if CH == "all": r = torch.cat([r, (it.float() / 127.5 - 1.0)], 1)
    return r


cls_freq = np.array([sum((S["L"].cls == c).sum() for S in SITES.values()) for c in CLASSES], float)
MIN_OBJ = int(os.environ.get("MIN_OBJ", 10))
ACTIVE = [c for c, n in zip(CLASSES, cls_freq) if n >= MIN_OBJ]
cls_w = torch.tensor([1.0 if c in ACTIVE else 0.0 for c in CLASSES], device=dev).view(1, -1, 1, 1)
GLOBAL_W = {c: (1.0 / n if c in ACTIVE else 0.0) for c, n in zip(CLASSES, cls_freq)}  # fixed rarity weights for the score
print("active classes:", ACTIVE, "counts:", dict(zip(CLASSES, cls_freq.astype(int))), flush=True)


_POOLS = {}


def sample_batch(train_sites, bs, n):
    D, I, M, V, Hm = [], [], [], [], []
    w_obj = 1 / np.sqrt(np.maximum(cls_freq, 1)) * np.array([c in ACTIVE for c in CLASSES])
    w_obj = np.minimum(w_obj, 4 * w_obj[w_obj > 0].min())  # cap rare-class oversampling at 4x
    key_ = tuple(train_sites)
    if key_ not in _POOLS:
        _POOLS[key_] = {ci: [(s_, i) for s_ in train_sites for i, cc in enumerate(SITES[s_]["L"].cls) if cc == CLASSES[ci]]
                        for ci in range(len(CLASSES)) if w_obj[ci] > 0}
        _POOLS[key_] = {k: v for k, v in _POOLS[key_].items() if v}
    pools = _POOLS[key_]
    for _ in range(bs):
        if random.random() < 0.6 and pools:
            # object-centred, class-balanced over classes that exist in THIS fold's training sites
            ks = list(pools); pw = np.array([w_obj[k] for k in ks]); c = ks[np.random.choice(len(ks), p=pw / pw.sum())]
            s, i = random.choice(pools[c]); (px, py), _ = SITES[s]["cen"][i]
            cy, cx = py + random.uniform(-n / 3, n / 3), px + random.uniform(-n / 3, n / 3)
        else:
            s = random.choice(train_sites); vv = SITES[s]["valid"]
            for _ in range(20):
                cy, cx = random.randrange(vv.shape[0]), random.randrange(vv.shape[1])
                if vv[cy, cx]: break
        d, it, mk, vd, hm = chip(s, cy, cx, n)
        k = random.randrange(4); fl = random.random() < 0.5
        aug = lambda a: np.ascontiguousarray(np.flip(np.rot90(a, k, axes=(-2, -1)), -1) if fl else np.rot90(a, k, axes=(-2, -1)))
        D.append(aug(d)); I.append(aug(it)); M.append(aug(mk)); V.append(aug(vd)); Hm.append(aug(hm))
    t = lambda a, dt: torch.from_numpy(np.stack(a)).to(dev, dt)
    return t(D, torch.float32)[:, None], t(I, torch.uint8)[:, None], t(M, torch.float32), t(V, torch.float32)[:, None], t(Hm, torch.float32)[:, None]


def dice_loss(p, y, v):
    inter = (p * y * v).sum((2, 3)); den = ((p + y) * v).sum((2, 3))
    has = (y * v).sum((2, 3)) > 0
    d = 1 - (2 * inter + 1) / (den + 1)
    return (d * has).sum() / has.sum().clamp(min=1)


def tversky_loss(p, y, v, a=0.3, b=0.7):
    tp = (p * y * v).sum((2, 3)); fp = (p * (1 - y) * v).sum((2, 3)); fn = ((1 - p) * y * v).sum((2, 3))
    has = (y * v).sum((2, 3)) > 0
    t = 1 - (tp + 1) / (tp + a * fp + b * fn + 1)
    return (t * has).sum() / has.sum().clamp(min=1)


def seg_loss(logit, y, v):
    p = torch.sigmoid(logit)
    if LOSS == "bce":
        return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * v).mean()
    if LOSS == "bce_dice":
        bce = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * v).mean()
        return bce + dice_loss(p, y, v)
    focal = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * ((1 - p) * y + p * (1 - y)) ** 2 * v).mean() * 10
    return focal + (tversky_loss(p, y, v) if LOSS == "tversky" else dice_loss(p, y, v))


def heat_loss(logit, hm, v):  # CenterNet penalty-reduced focal loss
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4); pos = (hm > 0.99).float()
    l = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * (1 - hm) ** 4 * p ** 2 * torch.log(1 - p)) * v
    return l.sum() / pos.sum().clamp(min=1)


def make_model():
    kw = dict(encoder_name=ENC, encoder_weights="imagenet", in_channels=IN_CH, classes=len(CLASSES) + HEAT)
    kw.update(json.loads(os.environ.get("ENC_KW", "{}")))  # e.g. Swin: {"img_size": 448, "strict_img_size": false}
    m = getattr(smp, ARCH)(**kw)
    head = [mod for mod in m.segmentation_head.modules() if isinstance(mod, nn.Conv2d)][-1]
    with torch.no_grad():  # focal-style prior init: p(class)=0.1, p(centre)=0.01
        head.bias[:len(CLASSES)].fill_(-2.2)
        if HEAT: head.bias[len(CLASSES):].fill_(-4.6)
    return m.to(dev)


def model_cost(model):
    from torch.utils.flop_counter import FlopCounterMode
    n_par = sum(p.numel() for p in model.parameters())
    x = torch.zeros(1, IN_CH, 512, 512, device=dev)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return n_par, fc.get_total_flops()


@torch.no_grad()
def predict_site(model, site, n=int(os.environ.get("PRED", 512)), ov=128):
    S = SITES[site]; H, W = S["valid"].shape; step = n - ov
    acc = np.zeros((len(CLASSES) + HEAT, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    w1 = np.hanning(n + 2)[1:-1]; win = np.outer(w1, w1).astype(np.float32) + 1e-3
    tiles = [(y, x) for y in range(0, max(H - ov, 1), step) for x in range(0, max(W - ov, 1), step)]
    t0 = time.time()
    for k in range(0, len(tiles), 8):
        batch = tiles[k:k + 8]; D, I = [], []
        for (y, x) in batch:
            d, it, _, _, _ = chip(site, y + n // 2, x + n // 2, n); D.append(d); I.append(it)
        dt = torch.from_numpy(np.stack(D)).to(dev)[:, None]; itt = torch.from_numpy(np.stack(I)).to(dev)[:, None]
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            pr = torch.sigmoid(model(to_input(dt, itt)).float()).cpu().numpy()
        for (y, x), p in zip(batch, pr):
            hh, ww = min(n, H - y), min(n, W - x)
            acc[:, y:y + hh, x:x + ww] += p[:, :hh, :ww] * win[:hh, :ww]; wsum[y:y + hh, x:x + ww] += win[:hh, :ww]
    acc /= np.maximum(wsum, 1e-6); acc[:, ~S["valid"]] = 0
    return acc, time.time() - t0


# ------------------------------------------------------------------ train / OOF
all_pred, all_gt, summary = [], [], []
pix = [[0, 0] for _ in CLASSES]  # pixel intersection/union per class over all OOF sites
for fold in FOLDS:
    tr = [s for s in SITES if gl.set_index("site").fold[s] != fold]; va = [s for s in SITES if gl.set_index("site").fold[s] == fold]
    if not va: continue
    if not tr and SMOKE: tr = va  # single-site local smoke only
    model = make_model()
    if fold == FOLDS[0]:
        n_par, flops = model_cost(model); print(f"params={n_par/1e6:.2f}M  FLOPs@512={flops/1e9:.1f}G", flush=True)
    enc_p = [p for n_, p in model.named_parameters() if n_.startswith("encoder")]
    oth_p = [p for n_, p in model.named_parameters() if not n_.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": LR * 0.5}, {"params": oth_p, "lr": LR}], weight_decay=1e-4)
    total = EPOCHS * ITERS if not SMOKE else 30
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR * 0.5, LR], total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda"); t0 = time.time(); model.train()
    for it_ in range(total):
        d, itn, y, v, hm = sample_batch(tr, BS, CHIP)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(to_input(d, itn))
        out = out.float()
        loss = seg_loss(out[:, :len(CLASSES)], y, v * cls_w) + (heat_loss(out[:, -1:], hm, v) if HEAT else 0)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        if it_ % 250 == 0: print(f"fold {fold} it {it_}/{total} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), f"{OUT}/model_f{fold}.pt")
    model.eval(); km2 = 0.0; tsec = 0.0
    for s in va:
        acc, dt = predict_site(model, s); S = SITES[s]; tsec += dt; km2 += S["valid"].sum() * GSD * GSD / 1e6
        heat = acc[-1] if HEAT else None
        for thr in (0.3, 0.4, 0.5):
            for c, poly, sc in INS.extract(acc[:len(CLASSES)], heat, S["T"], CLASSES, thr=thr, gsd=GSD):
                if c not in ACTIVE: continue
                all_pred.append(dict(group=s, cls=c, geometry=poly, score=sc, thr=thr, fold=fold))
        for c, g in zip(S["L"].cls, S["L"].geometry): all_gt.append(dict(group=s, cls=c, geometry=g, fold=fold))
        np.save(f"{OUT}/prob_{s}.npy", (acc * 255).astype(np.uint8))
        gtm = np.stack([(S["mask"] >> ci) & 1 for ci in range(len(CLASSES))]).astype(bool)
        for ci in range(len(CLASSES)):
            pm = (acc[ci] >= 0.5) & S["valid"]; gm = gtm[ci] & S["valid"]
            pix[ci][0] += int((pm & gm).sum()); pix[ci][1] += int((pm | gm).sum())
    gpd.GeoDataFrame(all_pred).to_pickle(f"{OUT}/oof_pred_partial.pkl"); gpd.GeoDataFrame(all_gt).to_pickle(f"{OUT}/oof_gt_partial.pkl")
    json.dump(dict(done_folds=[f["fold"] for f in summary] + [fold], pix_counts=pix), open(f"{OUT}/partial.json", "w"))
    summary.append(dict(fold=fold, n_val_sites=len(va), train_min=round((time.time() - t0) / 60, 1), infer_s_per_km2=round(tsec / max(km2, 1e-6), 1)))
    print(summary[-1], flush=True)

P = gpd.GeoDataFrame(all_pred); G = gpd.GeoDataFrame(all_gt)
P.to_pickle(f"{OUT}/oof_pred.pkl"); G.to_pickle(f"{OUT}/oof_gt.pkl")
res = []
for thr in (0.3, 0.4, 0.5):
    Tm = MET.evaluate(P[P.thr == thr], G, CLASSES, thresholds={c: 0.0 for c in CLASSES}, class_weights=GLOBAL_W)
    Pk = P[(P.thr == thr) & P.cls.str.startswith("курганы")].assign(cls="курганы"); Gk = G[G.cls.str.startswith("курганы")].assign(cls="курганы")
    Tk = MET.evaluate(Pk, Gk, ["курганы"], thresholds={"курганы": 0.0}).iloc[:1].assign(cls="KURGAN_ANY")
    Tm = pd.concat([Tm, Tk], ignore_index=True)
    Tm.insert(0, "mask_thr", thr); res.append(Tm)
R = pd.concat(res); R.to_csv(f"{OUT}/metrics.csv", index=False)
# chance baseline per class/site at the 0.4 operating point: same number, same areas, random centres
rng = np.random.default_rng(0); chance = []
from shapely.geometry import Point
for s in G.group.unique():
    S = SITES[s]; vy, vx = np.nonzero(S["valid"][::8, ::8])
    for c in ACTIVE:
        Pc = P[(P.group == s) & (P.cls == c) & (P.thr == 0.4)]
        if not len(Pc): continue
        k = rng.integers(0, len(vy), len(Pc))
        pts = [S["T"] * (vx[i] * 8 + 4, vy[i] * 8 + 4) for i in k]
        for (x, y), a in zip(pts, Pc.geometry.area): chance.append(dict(group=s, cls=c, geometry=Point(x, y).buffer(math.sqrt(a / math.pi)), score=1.0))
Ch = MET.evaluate(gpd.GeoDataFrame(chance) if chance else gpd.GeoDataFrame(columns=["group", "cls", "geometry", "score"]), G, CLASSES, thresholds={c: 0.5 for c in CLASSES}, class_weights=GLOBAL_W)
Ch.to_csv(f"{OUT}/chance.csv", index=False)
iou = {c: (pix[i][0] / pix[i][1] if pix[i][1] else float("nan")) for i, c in enumerate(CLASSES) if c in ACTIVE}
miou = float(np.nanmean([v for v in iou.values()]))
pd.DataFrame([dict(cls=c, IoU=v) for c, v in iou.items()] + [dict(cls="mIoU", IoU=miou)]).to_csv(f"{OUT}/pixel_iou.csv", index=False)
print("pixel IoU", {k: round(v, 3) for k, v in iou.items()}, "mIoU", round(miou, 3))
meta = dict(tag=TAG, arch=ARCH, enc=ENC, loss=LOSS, ch=CH, heat=HEAT, epochs=EPOCHS, iters=ITERS, bs=BS, chip=CHIP,
            params_M=round(n_par / 1e6, 2), gflops_512=round(flops / 1e9, 1), mIoU=round(miou, 4), pix_counts=pix, active=ACTIVE, global_w=GLOBAL_W, folds=summary)
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1, ensure_ascii=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string()); print("CHANCE\n", Ch.round(3).to_string()); print(json.dumps(meta, ensure_ascii=False)); print("DONE")
