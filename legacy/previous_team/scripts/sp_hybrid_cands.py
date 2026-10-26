"""Satellite hybrid, stage 1: physics candidates + hand-crafted + frozen-CNN features (CPU-efficient).

    python scripts/sp_hybrid_cands.py [--scenes all|N] [--out work/sp_hybrid]

Why: a U-Net trained end-to-end on this CPU underfits (run sp_r18l_v1: official honest 0.047). Objects here are mostly
round soil/colour marks (kurgans), compact blocks (architecture) or big enclosures (hillforts), so a scale-space blob
detector proposes them well; a gradient-boosted model then scores each candidate from cheap features.

Per scene (work/sp/img/*.npy, 0.5 m RGB + valid), processed in 2048 px tiles with a 300 px overlap (memory-bounded):
  * CIELAB conversion; scale-normalised Laplacian-of-Gaussian of L at radii 3, 5, 8, 12 m (0.5 m grid) and
    18, 27, 40, 60 m (1 m grid), and of the a/b chroma at 5, 12, 27 m; candidates = local maxima of max |LoG| over
    scales (light and dark blobs), >= ~2 m apart, top-K per scene (K = 600 per km^2, at least 300).
  * features: LoG responses at the centre, best scale/sign; disc-vs-ring colour contrast (L, a, b) and gradient at the
    best radius; texture; frozen ImageNet ResNet-18 layer-3 embedding (256-d, global-average pooled) of a crop of side
    4r resized to 64 px; scene brightness context and candidate density.
  * targets: the official matching rule applied to the candidate disc (radius r): positive for the class of the label
    it would hit (centroid inside AND coverage >= 0.5 of the smaller shape), one-to-one, strongest blob first.
Output: <out>/parts/<scene>.parquet (resumable cache), <out>/cands.parquet, <out>/recall_ceiling.csv."""
import argparse, json, math, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import cv2, torch
from scipy import ndimage as ndi
from shapely import wkb, STRtree
from shapely.geometry import Point
from rasterio.transform import Affine
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser(); ap.add_argument("--scenes", default="all"); ap.add_argument("--out", default="work/sp_hybrid")
ap.add_argument("--dens", type=float, default=600.0); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--tile", type=int, default=2048)
ap.add_argument("--unet", default="work/sp_runs/sp_r18l_v1", help="dir with out-of-fold prob_<scene>.npy; empty to disable")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(a.threads)
CLASSES = ["курганы_целые", "курганы_поврежденные", "городища", "фортификации", "архитектуры"]
R_FINE = (3, 5, 8, 12); R_COARSE = (18, 27, 40, 60); GSD = 0.5
TILE, PAD = a.tile, 300  # 0.5 m px; PAD covers the largest LoG support (r = 60 m) and the 1.7 r ring
SC = pd.read_csv("work/sp/scenes.csv"); LB = pd.read_parquet("work/sp/labels.parquet")
folds = json.load(open("work/sp_runs/sp_r18l_v1/folds.json", encoding="utf-8"))

import segmentation_models_pytorch as smp
_enc = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1).encoder.eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255; STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


@torch.no_grad()
def embed(crops):
    x = (torch.from_numpy(np.stack(crops)).float() - MEAN) / STD
    return _enc(x)[3].mean((2, 3)).numpy().astype(np.float32)  # layer3 (stride 16, 256 ch), global average


def log_stack(ch, radii, gsd):
    out = []
    for r in radii:
        s = max(r / math.sqrt(2) / gsd, 0.7)
        out.append((-ndi.gaussian_laplace(ch, s) * s * s).astype(np.float32))  # positive on bright blobs
    return np.stack(out)


def window_features(rgb, valid, T, K, core, unet=None):
    """Candidates and features for one window; keeps only candidates whose pixel lies in core = (y0, x0, y1, x1).
    unet: optional (6, H, W) uint8 out-of-fold U-Net probabilities on the same grid (stacking: extra candidates at its
    peaks and its class probabilities as features)."""
    H, W = valid.shape
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); lab[~valid] = 0
    L = lab[..., 0]; A_ = lab[..., 1] - 128; B_ = lab[..., 2] - 128
    del lab
    Lm = np.where(valid, L, np.median(L[valid]) if valid.any() else 0).astype(np.float32)
    fine = log_stack(Lm, R_FINE, GSD)
    coarse = log_stack(cv2.resize(Lm, (max(W // 2, 1), max(H // 2, 1)), interpolation=cv2.INTER_AREA), R_COARSE, 2 * GSD)
    cs = np.maximum(np.abs(fine).max(0), cv2.resize(np.abs(coarse).max(0), (W, H), interpolation=cv2.INTER_NEAREST))
    y0, x0, y1, x1 = core
    keep = np.zeros_like(valid); keep[y0:y1, x0:x1] = True; cs[~(valid & keep)] = 0
    mx = ndi.maximum_filter(cs, size=int(2 * 0.7 * 3 / GSD) + 1)
    pk = np.argwhere((cs == mx) & (cs > 0))
    if len(pk) > K:
        pk = pk[np.argsort(-cs[pk[:, 0], pk[:, 1]])[:K]]
    src = np.zeros(len(pk), np.int8)
    if unet is not None:  # U-Net peaks (max over object classes) as extra candidates
        um = ndi.gaussian_filter(unet[:5].max(0).astype(np.float32) / 255, 2.0); um[~(valid & keep)] = 0
        upk = np.argwhere((um == ndi.maximum_filter(um, size=25)) & (um >= 0.25))  # >= 6 m apart; background is 0.1-0.3
        if len(upk) > K:
            upk = upk[np.argsort(-um[upk[:, 0], upk[:, 1]])[:K]]
        if len(upk):
            pk = np.concatenate([pk, upk]); src = np.concatenate([src, np.ones(len(upk), np.int8)])
    del cs, mx, keep
    rr, cc_ = pk[:, 0], pk[:, 1]
    F = {"src": src}
    for i, rad in enumerate(R_FINE):
        F[f"log_L_{rad}"] = fine[i][rr, cc_]
    for i, rad in enumerate(R_COARSE):
        F[f"log_L_{rad}"] = coarse[i][np.minimum(rr // 2, coarse.shape[1] - 1), np.minimum(cc_ // 2, coarse.shape[2] - 1)]
    del fine, coarse
    allr = np.array(R_FINE + R_COARSE, float)
    M = np.stack([F[f"log_L_{x}"] for x in R_FINE + R_COARSE], 1) if len(rr) else np.zeros((0, len(allr)), np.float32)
    best = np.abs(M).argmax(1) if len(rr) else np.zeros(0, int)
    F["best_r"] = allr[best]
    F["best_sign"] = np.sign(M[np.arange(len(M)), best]) if len(rr) else np.zeros(0)
    F["best_abs"] = np.abs(M).max(1) if len(rr) else np.zeros(0)
    for nm, ch in (("A", A_), ("B", B_)):
        st = log_stack(ch, (5, 12, 27), GSD)
        for i, rad in enumerate((5, 12, 27)):
            F[f"log_{nm}_{rad}"] = st[i][rr, cc_]
        del st
    gm = cv2.magnitude(cv2.Sobel(Lm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lm, cv2.CV_32F, 0, 1, ksize=3))
    ii = {k: cv2.integral(v.astype(np.float64)) for k, v in (("L", L), ("A", A_), ("B", B_), ("L2", L * L), ("V", valid.astype(np.float64)), ("G", gm))}
    del gm

    def boxsum(I, y0, x0, y1, x1):
        y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H); x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
        return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
    rp = np.maximum(F["best_r"] / GSD, 2).astype(int); ro = (rp * 1.7).astype(int)
    d0 = (rr - rp, cc_ - rp, rr + rp + 1, cc_ + rp + 1); d1 = (rr - ro, cc_ - ro, rr + ro + 1, cc_ + ro + 1)
    nin = np.maximum(boxsum(ii["V"], *d0), 1); nout = np.maximum(boxsum(ii["V"], *d1) - boxsum(ii["V"], *d0), 1)
    for k in ("L", "A", "B", "G"):
        sin = boxsum(ii[k], *d0); sall = boxsum(ii[k], *d1)
        F[f"in_{k}"] = sin / nin; F[f"ring_{k}"] = (sall - sin) / nout; F[f"diff_{k}"] = F[f"in_{k}"] - F[f"ring_{k}"]
    F["std_in_L"] = np.sqrt(np.maximum(boxsum(ii["L2"], *d0) / nin - F["in_L"] ** 2, 0))
    F["valid_frac"] = nin / ((2 * rp + 1) ** 2)
    if unet is not None:  # out-of-fold U-Net class probabilities: centre value and disc mean
        for ci in range(unet.shape[0]):
            ch = unet[ci].astype(np.float64) / 255; Ic = cv2.integral(ch)
            F[f"un_c{ci}"] = ch[rr, cc_]; F[f"un_m{ci}"] = boxsum(Ic, *d0) / ((2 * rp + 1) ** 2)
    del ii
    crops = []
    for y, x, rpx in zip(rr, cc_, rp):
        h = int(max(16, 2 * rpx)); yy0, xx0 = y - h, x - h
        c = np.zeros((2 * h, 2 * h, 3), np.uint8)
        ys, xs = slice(max(yy0, 0), min(yy0 + 2 * h, H)), slice(max(xx0, 0), min(xx0 + 2 * h, W))
        c[ys.start - yy0:ys.stop - yy0, xs.start - xx0:xs.stop - xx0] = rgb[ys, xs]
        crops.append(np.moveaxis(cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA), -1, 0))
    E = np.concatenate([embed(crops[i:i + 256]) for i in range(0, len(crops), 256)]) if crops else np.zeros((0, 256), np.float32)
    for j in range(E.shape[1]):
        F[f"cnn{j}"] = E[:, j]
    D = pd.DataFrame(F)
    D["x"] = T.c + (cc_ + 0.5) * T.a; D["y"] = T.f + (rr + 0.5) * T.e
    return D


def scene_candidates(sid):
    r = SC[SC.scene == sid].iloc[0]; T = Affine(*json.loads(r["T"]))
    arr = np.load(f"work/sp/img/{sid}.npy", mmap_mode="r")
    H, W = arr.shape[1:]
    sub = np.asarray(arr[:, ::4, ::4]); vs = sub[3] > 0
    Ls = cv2.cvtColor(np.ascontiguousarray(np.moveaxis(sub[:3], 0, -1)), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    scene_L, scene_Ls = (float(Ls[vs].mean()), float(Ls[vs].std())) if vs.any() else (0.0, 0.0)
    km2 = float(vs.sum()) * 16 * GSD * GSD / 1e6; K = int(max(300, a.dens * km2))
    up = f"{a.unet}/prob_{sid}.npy" if a.unet else None
    UN = np.load(up, mmap_mode="r") if up and os.path.exists(up) else None
    parts = []
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            wy0, wx0 = max(ty - PAD, 0), max(tx - PAD, 0); wy1, wx1 = min(ty + TILE + PAD, H), min(tx + TILE + PAD, W)
            v = np.asarray(arr[3, wy0:wy1, wx0:wx1]) > 0
            core = (ty - wy0, tx - wx0, min(ty + TILE, H) - wy0, min(tx + TILE, W) - wx0)
            if not v[core[0]:core[2], core[1]:core[3]].any():
                continue
            rgb = np.ascontiguousarray(np.moveaxis(np.asarray(arr[:3, wy0:wy1, wx0:wx1]), 0, -1))
            uw = np.asarray(UN[:, wy0:wy1, wx0:wx1]) if UN is not None else None
            parts.append(window_features(rgb, v, T * Affine.translation(wx0, wy0), K, core, uw))
            del rgb, v
    D = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(D) and (D.src == 0).sum() > K:  # cap the blob candidates only; U-Net peaks are kept
        D = pd.concat([D[D.src == 0].nlargest(K, "best_abs"), D[D.src == 1]], ignore_index=True)
    D["scene_L"] = scene_L; D["scene_Lstd"] = scene_Ls; D["dens_km2"] = len(D) / max(km2, 1e-3); D["r_m"] = D.best_r
    L_ = LB[LB.scene == sid]; gts = [(c, wkb.loads(g)) for c, g in zip(L_.cls, L_.wkb)]
    D["tgt"] = 0  # 0 = background, 1..5 = CLASSES index + 1
    hit = 0
    if gts and len(D):
        xs_, ys_, rm = D.x.values, D.y.values, D.r_m.values
        tree = STRtree([g for _, g in gts]); used = set(); tcol = D.columns.get_loc("tgt")
        for i in np.argsort(-D.best_abs.values):
            c = Point(xs_[i], ys_[i]); disc = None; best, bj = -1.0, -1
            for j in tree.query(c):
                if j in used:
                    continue
                g = gts[j][1]
                if not g.contains(c):
                    continue
                disc = disc if disc is not None else c.buffer(rm[i], 16)
                cov = disc.intersection(g).area / max(min(disc.area, g.area), 1e-9)
                if cov >= 0.5 and cov > best:
                    best, bj = cov, j
            if bj >= 0:
                used.add(bj); D.iat[i, tcol] = CLASSES.index(gts[bj][0]) + 1
        hit = len(used)
    D["scene"] = sid; D["site"] = r.site; D["fold"] = folds.get(sid, -1)
    return D, len(gts), hit


t0 = time.time(); parts = []
os.makedirs(f"{a.out}/parts", exist_ok=True)
scenes = list(SC[SC.valid_frac > 0.01].scene)
if a.scenes != "all":
    scenes = scenes[:int(a.scenes)]
for k, sid in enumerate(scenes):
    pp = f"{a.out}/parts/{sid}.parquet"
    if os.path.exists(pp):  # resume: scene already done
        parts.append(pd.read_parquet(pp)); continue
    try:
        D, ng, hit = scene_candidates(sid)
    except Exception as e:  # MemoryError, OpenCV out-of-memory, anything: log and continue
        print("FAILED", sid, repr(e)[:200], flush=True); continue
    D.to_parquet(pp); parts.append(D)
    print(f"[{k + 1}/{len(scenes)}] {sid}: {len(D)} cands, labels {ng}, reachable {hit}, {time.time() - t0:.0f}s", flush=True)
C = pd.concat(parts, ignore_index=True); C.to_parquet(f"{a.out}/cands.parquet")
LB2 = LB[LB.scene.isin(set(C.scene))]
R = pd.DataFrame([dict(cls=c, n_gt=int((LB2.cls == c).sum()), reachable=int((C.tgt == i + 1).sum())) for i, c in enumerate(CLASSES)]).set_index("cls")
R["ceiling"] = (R.reachable / R.n_gt.clip(lower=1)).round(3)
R.to_csv(f"{a.out}/recall_ceiling.csv"); print(R); print(f"done {len(C)} candidates from {C.scene.nunique()} scenes in {time.time() - t0:.0f}s")
