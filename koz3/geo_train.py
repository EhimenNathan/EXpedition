"""Geophysics expert: U-Net with 2 heads on 0.1 m maps — finds heatmap (CenterNet focal) + object_poly mask.

Out-of-fold by region (FOLDS); finds loss only on MagGr maps that carry finds labels.
  python -m koz3.geo_train --fold 0        # train one fold (fold -1 = all regions, for deployment)
  python -m koz3.geo_train --oof           # predict + score every fold's held-out maps
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3.labels import DATA  # noqa: E402

PREP = os.path.join(DATA, "..", "geo_prep")
TAG = os.environ.get("GEO_TAG", "")
RUNS = os.path.join(DATA, "..", "..", "work", "geo")
CROP = 384
SIGMA = 3.0                      # px at 0.1 m → 0.3 m
# region groups → folds (sites kept together: Semibratnee 201-207 split by plot, Gnezdovo by plot)
FOLDS = {
    0: ["200", "205", "212"],
    1: ["201", "206", "209", "213"],
    2: ["202", "207", "210", "214", "217"],
    3: ["203", "208", "211", "215"],
    4: ["204", "216"],
}


def fold_of(region):
    return next(k for k, v in FOLDS.items() if region[:3] in v)


GEO_INPUT = os.environ.get("GEO_INPUT", "detrend")


def geo_input(img, valid, mode=None):
    """(3,H,W) uint8 map → network input. 'rgb': per-map standardised colours. 'detrend': luminance + two
    high-pass layers (lum − Gauss 1 m, lum − Gauss 3 m), robust-scaled over valid pixels — removes the smooth
    regional field that hides anomalies on interpolated maps (Gnezdovo terraces)."""
    from scipy import ndimage as ndi
    mode = mode or GEO_INPUT
    x = img.astype(np.float32)
    if mode == "rgb":
        mu = x[:, valid].mean(1, keepdims=True); sd = x[:, valid].std(1, keepdims=True) + 1e-3
        return (((x.reshape(3, -1) - mu) / sd).reshape(x.shape) * valid).astype(np.float32)
    lum = x.mean(0)
    fill = np.where(valid, lum, np.median(lum[valid]) if valid.any() else 0)
    chans = [fill]
    for s_m in (1.0, 3.0):
        chans.append(fill - ndi.gaussian_filter(fill, s_m / 0.1))
    out = []
    for c in chans:
        v = c[valid]
        med = np.median(v) if v.size else 0.0
        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-3 if v.size else 1.0
        out.append(np.clip((c - med) / mad, -8, 8) * valid)
    return np.stack(out).astype(np.float32)


def load_all():
    items = []
    for f in sorted(glob.glob(os.path.join(PREP, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        valid = z["valid"].astype(bool)
        if valid.sum() < 100:
            continue
        img = geo_input(z["img"], valid)
        items.append(dict(name=os.path.basename(f)[:-4], img=img.astype(np.float16), obj=z["obj"], valid=valid,
                          finds=z["finds"], has_finds=bool(z["has_finds"]), meta=meta, fold=fold_of(meta["region"])))
    return items


def heatmap(shape, pts, sigma=SIGMA):
    H, W = shape
    hm = np.zeros((H, W), np.float32)
    r = int(3 * sigma)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        x0, x1, y0, y1 = max(0, xi - r), min(W, xi + r + 1), max(0, yi - r), min(H, yi + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g[y0 - yi + r:y1 - yi + r, x0 - xi + r:x1 - xi + r])
    return hm


class Sampler:
    def __init__(self, items, rng):
        self.items, self.rng = items, rng
        for it in self.items:
            it["hm"] = heatmap(it["valid"].shape, it["finds"]) if it["has_finds"] else None
        w = np.array([np.sqrt(it["valid"].sum()) * (2.0 if it["meta"]["method"] == "MagGr" else 1.0) for it in items])
        self.p = w / w.sum()

    def crop(self):
        it = self.items[self.rng.choice(len(self.items), p=self.p)]
        H, W = it["valid"].shape
        c = CROP
        img = np.zeros((3, c, c), np.float32); obj = np.zeros((c, c), np.float32)
        val = np.zeros((c, c), np.float32); hm = np.zeros((c, c), np.float32)
        for _ in range(10):
            y = self.rng.integers(-c // 4, max(1, H - 3 * c // 4)); x = self.rng.integers(-c // 4, max(1, W - 3 * c // 4))
            ys, xs = slice(max(0, y), min(H, y + c)), slice(max(0, x), min(W, x + c))
            if it["valid"][ys, xs].mean() > 0.3 if it["valid"][ys, xs].size else False:
                break
        dy, dx = ys.start - y, xs.start - x
        h, w = ys.stop - ys.start, xs.stop - xs.start
        img[:, dy:dy + h, dx:dx + w] = it["img"][:, ys, xs]
        obj[dy:dy + h, dx:dx + w] = it["obj"][ys, xs]
        val[dy:dy + h, dx:dx + w] = it["valid"][ys, xs]
        has = it["hm"] is not None
        if has:
            hm[dy:dy + h, dx:dx + w] = it["hm"][ys, xs]
        k = self.rng.integers(4); fl = self.rng.random() < 0.5
        arrs = [np.rot90(a, k, axes=(-2, -1)) for a in (img, obj, val, hm)]
        if fl:
            arrs = [a[..., ::-1] for a in arrs]
        img, obj, val, hm = [np.ascontiguousarray(a) for a in arrs]
        img = img * self.rng.uniform(0.7, 1.3) + self.rng.normal(0, 0.2)
        if self.rng.random() < 0.3:
            img = img + self.rng.normal(0, 0.15, img.shape)
        return img, obj, val, hm, float(has)

    def batch(self, n):
        b = [self.crop() for _ in range(n)]
        return [torch.from_numpy(np.stack([x[i] for x in b]).astype(np.float32)) for i in range(5)]


def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, classes=2)


def focal_hm(logit, hm, mask):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = hm.ge(0.99).float()
    pl = -torch.log(p) * (1 - p) ** 2 * pos
    nl = -torch.log(1 - p) * p ** 2 * (1 - hm) ** 4 * (1 - pos)
    return ((pl + nl) * mask).sum() / pos.mul(mask).sum().clamp(min=1)


def seg_loss(logit, t, mask):
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    p = torch.sigmoid(logit) * mask
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * mask).sum() + 1)
    return bce + dice


def train(fold, iters=6000, bs=12, lr=3e-4, seed=0):
    items = load_all()
    tr = [it for it in items if it["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    S = Sampler(tr, rng)
    dev = "cuda"
    model = build_model().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    for i in range(iters):
        img, obj, val, hm, has = [x.to(dev, non_blocking=True) for x in S.batch(bs)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img.to(memory_format=torch.channels_last)).float()
        fm = val * has[:, None, None]
        loss = focal_hm(out[:, 0], hm, fm) + seg_loss(out[:, 1], obj, val)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if i % 500 == 0 or i == iters - 1:
            print(f"fold {fold} it {i} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(RUNS, f"geo{TAG}_f{fold}.pt"))


@torch.no_grad()
def predict_map(model, img, tile=768, overlap=128, tta=True):
    """img (3,H,W) float → (2,H,W) sigmoid probabilities, tiled with flip TTA."""
    dev = next(model.parameters()).device
    C, H, W = img.shape
    out = np.zeros((2, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    step = tile - overlap
    for y in range(0, max(1, H - overlap), step):
        for x in range(0, max(1, W - overlap), step):
            y1, x1 = min(H, y + tile), min(W, x + tile)
            t = img[:, y:y1, x:x1]
            ph, pw = (-t.shape[1]) % 32, (-t.shape[2]) % 32
            tt = torch.from_numpy(np.pad(t, ((0, 0), (0, ph), (0, pw)))).float()[None].to(dev)
            variants = [(False, False), (True, False), (False, True)] if tta else [(False, False)]
            acc = 0
            for fx, fy in variants:
                z = tt
                if fx: z = z.flip(-1)
                if fy: z = z.flip(-2)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = torch.sigmoid(model(z).float())
                if fx: o = o.flip(-1)
                if fy: o = o.flip(-2)
                acc = acc + o
            o = (acc / len(variants))[0, :, :t.shape[1], :t.shape[2]].cpu().numpy()
            out[:, y:y1, x:x1] += o; cnt[y:y1, x:x1] += 1
    return out / np.maximum(cnt, 1)


def peaks(hm, valid, thr=0.05, nms=7):
    t = torch.from_numpy(hm)[None, None]
    mx = F.max_pool2d(t, nms, 1, nms // 2)
    keep = ((t == mx) & (t > thr))[0, 0].numpy() & valid
    ys, xs = np.nonzero(keep)
    return xs, ys, hm[ys, xs]


def objects_from_mask(prob, valid, thr=0.5, min_area_px=150):
    """Mask → list of (shapely polygon in pixel coords, score)."""
    import cv2
    from shapely.geometry import Polygon
    m = ((prob > thr) & valid).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    res = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area_px:
            continue
        comp = (lab == k).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        if len(c) < 4:
            continue
        poly = Polygon(c[:, 0, :].astype(float) + 0.5).buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        res.append((poly.simplify(1.0), float(prob[lab == k].mean())))
    return res


def to_3857_fn(meta):
    from pyproj import Transformer
    from rasterio.transform import Affine
    t = Affine(*meta["transform"])
    tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True)

    def f(x, y):
        X, Y = t * (np.asarray(x, float), np.asarray(y, float))
        return tr.transform(X, Y)
    return f


def eval_maps(items):
    """One map per (region, key) as validation provides: MagGr grey preferred over colour duplicates."""
    keep = []
    for it in items:
        m = it["meta"]
        if m["method"] == "MagGr" and m["key"] == "цвет":
            continue
        keep.append(it)
    return keep


def oof_predict(folds=FOLDS):
    items = load_all()
    dev = "cuda"
    cache = {}
    for fold in folds:
        model = build_model(pretrained=False).to(dev).eval()
        sd = torch.load(os.path.join(RUNS, f"geo{TAG}_f{fold}.pt"), map_location=dev)
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        for it in eval_maps([i for i in items if i["fold"] == fold]):
            cache[it["name"]] = (predict_map(model, it["img"].astype(np.float32)), it)
    return cache


def make_preds(cache, f_thr=0.1, o_thr=0.5, o_min_m2=1.5, r_find=0.5):
    from shapely.geometry import Polygon, Point
    from shapely.ops import transform as stx
    feats = []
    for name, (prob, it) in cache.items():
        f = to_3857_fn(it["meta"])
        if it["meta"]["method"] == "MagGr":
            xs, ys, sc = peaks(prob[0], it["valid"], f_thr)
            X, Y = f(xs + 0.5, ys + 0.5)
            k = 1 / np.cos(np.radians(_lat(Y)))   # 3857 scale factor
            for x, y, s, kk in zip(X, Y, sc, k):
                feats.append({"type": "Feature", "geometry": Point(x, y).buffer(r_find * kk, 8).__geo_interface__,
                              "properties": {"class_name": "finds_points", "confidence": float(s), "region_name": it["meta"]["region"]}})
        for poly, s in objects_from_mask(prob[1], it["valid"], o_thr, int(o_min_m2 / 0.01)):
            g = stx(lambda x, y, z=None: f(x, y), poly)
            feats.append({"type": "Feature", "geometry": g.__geo_interface__,
                          "properties": {"class_name": "object_poly", "confidence": s, "region_name": it["meta"]["region"]}})
    return {"type": "FeatureCollection", "features": feats}


def _lat(Y):
    return np.degrees(2 * np.arctan(np.exp(np.asarray(Y) / 6378137.0)) - np.pi / 2)


def gt_for(regions):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    return to_gt_geojson(L[L.region.isin(regions) & L["mod"].isin(["MagGr", "EtGr", "TerFl"])])


def oof_report(cache):
    from koz3.score import score
    regions = sorted({it["meta"]["region"] for _, it in cache.values()})
    gt = gt_for(regions)
    best = None
    for f_thr in (0.05, 0.1, 0.15, 0.2, 0.3):
        for o_thr in (0.3, 0.4, 0.5, 0.6):
            for o_min in (0.5, 1.5, 4.0):
                s, r = score(make_preds(cache, f_thr, o_thr, o_min), gt, ["finds_points", "object_poly"])
                row = (s, f_thr, o_thr, o_min, r["finds_points"]["f1"], r["object_poly"]["f1"])
                print("geo %.4f  f_thr %.2f o_thr %.2f o_min %.1f  finds %.4f  obj %.4f" % row, flush=True)
                best = max(best or row, row)
    print("BEST", best)
    # per-fold at the best shared setting
    _, f_thr, o_thr, o_min, *_ = best
    for fold, regs in FOLDS.items():
        sub = {k: v for k, v in cache.items() if v[1]["fold"] == fold}
        if not sub:
            continue
        rs = sorted({it["meta"]["region"] for _, it in sub.values()})
        s, r = score(make_preds(sub, f_thr, o_thr, o_min), gt_for(rs), ["finds_points", "object_poly"])
        print(f"fold {fold} {rs}: finds {r['finds_points']['f1']:.3f} (tp {r['finds_points']['tp']} fp {r['finds_points']['fp']} fn {r['finds_points']['fn']})"
              f"  obj {r['object_poly']['f1']:.3f} (tp {r['object_poly']['tp']} fp {r['object_poly']['fp']} fn {r['object_poly']['fn']})")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--oof", action="store_true")
    a = ap.parse_args()
    if a.fold is not None:
        train(a.fold, a.iters)
    if a.oof:
        import pickle
        cache = oof_predict()
        pickle.dump({k: (v[0].astype(np.float16), v[1]["name"]) for k, v in cache.items()}, open(os.path.join(RUNS, f"oof_probs{TAG}.pkl"), "wb"))
        oof_report(cache)
