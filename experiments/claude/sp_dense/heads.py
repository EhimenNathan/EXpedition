"""Satellite detection from scratch on CPU: a small dilated-conv head on frozen DINOv3-SAT token grids (extract.py),
trained per fold (koz3.sp_prep.site_folds, 20 km clusters) → out-of-fold maps per scene (work/sp_dense/oof/<idx>.npz).

Channels (token grid, 8 m at GSD 0.5):
  0 kurgan centre heat      CenterNet Gaussian at each kurgan's representative point, σ = max(4 m, 0.2·√area)
  1 damaged share           target 1 damaged / 0 intact, supervised only within kurgan footprints
  2 architecture centre heat
  3 fortification centre heat
  4 hillfort mask           token centre inside a hillfort polygon
Loss: penalty-reduced focal loss (CenterNet) on 0, 2, 3; BCE on 1 (masked) and 4 (pos-weighted); only valid tokens.

  python experiments/claude/sp_dense/heads.py train [iters=2500]   # all folds, writes OOF maps
"""
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, os.environ.get("SP_DENSE_OUT", "work/sp_dense"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")
MOD = os.environ.get("KOZ3_MOD", "SpOr")
HEAT = {"k": 0, "arkhitektury": 2, "fortifikatsii": 3}
NCH = 5
CROP = 48


def load_scene(path):
    from rasterio.transform import Affine
    z = np.load(path, allow_pickle=True)
    return dict(f=z["f"], T=Affine(*z["T"]), crs=str(z["crs"]), valid=z["valid"], region=str(z["region"]), tif=str(z["tif"]))


def targets(s, L):
    """Token-grid targets for one scene from the region's satellite labels (EPSG:3857 → scene CRS)."""
    gh, gw = s["f"].shape[:2]
    Y = np.zeros((NCH, gh, gw), np.float32); M = np.zeros((gh, gw), bool)       # M: damaged-share supervision mask
    vh, vw = s["valid"]; valid = np.zeros((gh, gw), bool); valid[:int(np.ceil(vh)), :int(np.ceil(vw))] = True
    lab = L[L.region == s["region"]]
    if len(lab):
        lab = lab.to_crs(s["crs"])
        inv = ~s["T"]; px = abs(s["T"].a)                                         # metres per token
        rr, cc = np.mgrid[0:gh, 0:gw]
        from shapely import contains_xy
        for g, c in zip(lab.geometry, lab.class_name):
            if g is None or g.is_empty:
                continue
            col, row = inv * g.representative_point().coords[0]
            col -= 0.5; row -= 0.5
            if not (-4 <= row < gh + 4 and -4 <= col < gw + 4):
                continue
            if c == "gorodishcha":
                X, Yc = s["T"] * (cc + 0.5, rr + 0.5)
                Y[4] = np.maximum(Y[4], contains_xy(g, X, Yc)); continue
            ch = HEAT["k"] if c in K else HEAT.get(c)
            if ch is None:
                continue
            sig = max(4.0, 0.2 * np.sqrt(g.area)) / px
            r0, r1 = max(0, int(row - 3 * sig) - 1), min(gh, int(row + 3 * sig) + 2)
            c0, c1 = max(0, int(col - 3 * sig) - 1), min(gw, int(col + 3 * sig) + 2)
            if r0 >= r1 or c0 >= c1:
                continue
            gy, gx = np.mgrid[r0:r1, c0:c1]
            Y[ch, r0:r1, c0:c1] = np.maximum(Y[ch, r0:r1, c0:c1], np.exp(-((gy - row) ** 2 + (gx - col) ** 2) / (2 * sig * sig)))
            ir, ic = int(round(row)), int(round(col))
            if 0 <= ir < gh and 0 <= ic < gw:
                Y[ch, ir, ic] = 1.0                                                  # exact peak at the nearest token
            if c in K:
                near = np.exp(-((gy - row) ** 2 + (gx - col) ** 2) / (2 * sig * sig)) > 0.5
                Y[1, r0:r1, c0:c1][near] = float(c == K[0]); M[r0:r1, c0:c1] |= near
    return Y, M, valid


def make_head(cin=128, w=96):
    import torch.nn as nn
    layers, c = [], cin
    for d in (1, 1, 2, 4, 8):
        layers += [nn.Conv2d(c, w, 3, padding=d, dilation=d), nn.GroupNorm(8, w), nn.GELU()]; c = w
    return nn.Sequential(*layers, nn.Conv2d(w, NCH, 1))


def focal(pred, gt, valid):
    import torch
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = (gt >= 0.999).float() * valid
    neg = (1 - pos) * valid
    lp = -torch.log(p) * (1 - p) ** 2 * pos
    ln = -torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    return (lp.sum() + ln.sum()) / pos.sum().clamp(min=1)


def loss_fn(out, Y, M, V):
    import torch
    import torch.nn.functional as F
    l = sum(focal(out[:, ch], Y[:, ch], V) for ch in (0, 2, 3))
    if M.any():
        l = l + F.binary_cross_entropy_with_logits(out[:, 1][M], Y[:, 1][M])
    l = l + (F.binary_cross_entropy_with_logits(out[:, 4], Y[:, 4], reduction="none", pos_weight=torch.tensor(10.0)) * V).sum() / V.sum().clamp(min=1)
    return l


def crops(scene_data, n, rng):
    """n random CROP×CROP crops; scenes sampled ∝ sqrt(size), crops centred on a labelled token half of the time."""
    w = np.array([np.sqrt(d[0].shape[0] * d[0].shape[1]) for d in scene_data]); w /= w.sum()
    X, Y, M, V = [], [], [], []
    for _ in range(n):
        f, y, m, v = scene_data[rng.choice(len(scene_data), p=w)]
        gh, gw = f.shape[:2]
        pos = np.argwhere(y[[0, 2, 3, 4]].max(0) > 0.5)
        if len(pos) and rng.random() < 0.5:
            cr, cc = pos[rng.integers(len(pos))]; r = int(np.clip(cr - rng.integers(CROP), 0, max(gh - CROP, 0))); c = int(np.clip(cc - rng.integers(CROP), 0, max(gw - CROP, 0)))
        else:
            r, c = rng.integers(max(gh - CROP, 0) + 1), rng.integers(max(gw - CROP, 0) + 1)
        fx = np.zeros((CROP, CROP, f.shape[2]), np.float16); yy = np.zeros((NCH, CROP, CROP), np.float32)
        mm = np.zeros((CROP, CROP), bool); vv = np.zeros((CROP, CROP), bool)
        ch_, cw_ = min(CROP, gh - r), min(CROP, gw - c)
        fx[:ch_, :cw_] = f[r:r + ch_, c:c + cw_]; yy[:, :ch_, :cw_] = y[:, r:r + ch_, c:c + cw_]
        mm[:ch_, :cw_] = m[r:r + ch_, c:c + cw_]; vv[:ch_, :cw_] = v[r:r + ch_, c:c + cw_]
        k = rng.integers(4)                                                    # flips of the token grid (context aug)
        if k & 1:
            fx, yy, mm, vv = fx[:, ::-1], yy[:, :, ::-1], mm[:, ::-1], vv[:, ::-1]
        if k & 2:
            fx, yy, mm, vv = fx[::-1], yy[:, ::-1], mm[::-1], vv[::-1]
        X.append(fx.transpose(2, 0, 1)); Y.append(yy); M.append(mm); V.append(vv)
    return [np.ascontiguousarray(np.stack(a)) for a in (X, Y, M, V)]


def predict(model, f):
    import torch
    with torch.inference_mode():
        x = torch.from_numpy(f.astype(np.float32).transpose(2, 0, 1)[None])
        o = torch.sigmoid(model(x))[0].numpy()
        o2 = torch.sigmoid(model(torch.flip(x, (3,))))[0].numpy()[:, :, ::-1]            # flip TTA on the token grid
    return (o + o2) / 2


def train_all(iters=2500, seed=0):
    import geopandas as gpd
    import torch
    from koz3.sp_prep import site_folds
    torch.set_num_threads(os.cpu_count()); torch.manual_seed(seed)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == MOD]
    fo = site_folds(L)
    files = sorted(glob.glob(os.path.join(D, "scenes", "*.npz")))
    S = [load_scene(p) for p in files]
    data = []
    for s in S:
        Y, M, V = targets(s, L); data.append((s["f"], Y, M, V))
    fold = np.array([fo[s["region"]] for s in S])
    print(f"{len(S)} scenes; positives: kurgan peaks {int(sum((d[1][0] >= 0.999).sum() for d in data))}, "
          f"arch {int(sum((d[1][2] >= 0.999).sum() for d in data))}, fort {int(sum((d[1][3] >= 0.999).sum() for d in data))}, "
          f"hillfort tokens {int(sum(d[1][4].sum() for d in data))}", flush=True)
    os.makedirs(os.path.join(D, "oof"), exist_ok=True); os.makedirs(os.path.join(D, "models"), exist_ok=True)
    for f in sorted(set(fold)) + [-1]:
        tr = [data[i] for i in range(len(S)) if fold[i] != f]
        rng = np.random.default_rng(seed + 1 + f)
        model = make_head(); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=iters, pct_start=0.1)
        t0 = time.time()
        for it in range(iters):
            X, Y, M, V = [torch.from_numpy(a) for a in crops(tr, 16, rng)]
            loss = loss_fn(model(X.float()), Y, M, V.float())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % 250 == 0:
                print(f"  fold {f} it {it} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        model.eval()
        torch.save(model.state_dict(), os.path.join(D, "models", f"head_f{f}_s{seed}.pt"))
        if f >= 0:
            for i in np.flatnonzero(fold == f):
                np.savez_compressed(os.path.join(D, "oof", os.path.basename(files[i])), p=predict(model, S[i]["f"]).astype(np.float16))
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_all(*[int(a) for a in sys.argv[2:3]])
