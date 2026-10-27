"""Experiment 6 feasibility: frozen DINOv3-SAT (ViT-L/16, SAT-493M) features as a satellite kurgan verifier, CPU.

Candidates: multi-scale LoG blobs on single-image satellite screenshots (as in the shading proxy); label = candidate
inside a satellite kurgan polygon. Each candidate → 64 m × 64 m chip resampled to 0.5 m (128 px, upsampled to 224 for
the ViT) → frozen DINOv3 CLS + mean patch token. Linear probe (logistic regression), leave-one-scene-out, against the
appearance-only baseline (LoG response, scale, polarity). Also reports seconds per chip on this CPU and the model's
fp16 size, for the 30-min runtime and 2 GB archive limits.

Requires an accepted DINOv3 licence on Hugging Face and HF_TOKEN (read) in the environment.
  KOZ3_RAW=data/train-expds-3 python experiments/claude/sp_dinov3_probe.py [max_scenes=20]
"""
import glob, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
D = os.environ.get("KOZ3_RAW", "data/train-expds-3"); OUT = os.environ.get("KOZ3_OUT", "work/dinov3_probe")
MODEL = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-sat493m")
GSD = 0.5; CHIP = 128


def scene_candidates(path, polys_3857):
    """LoG blob candidates at kurgan scales (3–12 m radius) → (rows, cols, response, scale, polarity, label)."""
    import rasterio
    from scipy import ndimage as ndi
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    from shapely.ops import transform as stf
    from pyproj import Transformer
    with rasterio.open(path) as ds:
        a = ds.read(indexes=list(range(1, min(3, ds.count) + 1))).astype(np.float32)
        T, crs = ds.transform, ds.crs
    G = a.mean(0); res = abs(T.a)
    to = Transformer.from_crs(3857, crs, always_xy=True).transform
    polys = [stf(to, g) for g in polys_3857]; tree = STRtree(polys)
    rows = []
    for rm in (3, 6, 12):
        s = max(1.0, rm / res / 1.414); L = -ndi.gaussian_laplace(G, s) * s * s
        for sign in (1, -1):
            Z = sign * L; mx = ndi.maximum_filter(Z, size=int(3 * s) | 1)
            for r, c in np.argwhere((Z == mx) & (Z > np.percentile(Z, 99))):
                rows.append((r, c, Z[r, c], rm, sign))
    rows = sorted(rows, key=lambda q: -q[2])[:600]
    out = []
    for r, c, z, rm, sign in rows:
        x, y = rasterio.transform.xy(T, r, c); p = Point(x, y)
        lab = any(polys[j].contains(p) for j in tree.query(p))
        out.append((x, y, z, rm, sign, int(lab)))
    return a, T, crs, res, out


def chip(a, T, res, x, y):
    """64 m × 64 m chip around (x, y) at 0.5 m → uint8 HWC 128 × 128."""
    from scipy import ndimage as ndi
    inv = ~T; c, r = inv * (x, y)
    half = 32.0 / res
    r0, r1, c0, c1 = int(r - half), int(r + half), int(c - half), int(c + half)
    H, W = a.shape[1:]
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        return None
    sub = a[:, r0:r1, c0:c1]
    z = (CHIP / sub.shape[1], CHIP / sub.shape[2])
    sub = np.stack([ndi.zoom(b, z, order=1) for b in sub])[:, :CHIP, :CHIP]
    if sub.shape[0] == 1:
        sub = np.repeat(sub, 3, 0)
    return np.clip(sub.transpose(1, 2, 0), 0, 255).astype(np.uint8)


def main(max_scenes=20):
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from shapely.geometry import shape
    from shapely.ops import transform as stf
    from pyproj import Transformer
    tok = os.environ.get("HF_TOKEN")
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(MODEL, token=tok)
    model = AutoModel.from_pretrained(MODEL, token=tok, torch_dtype=torch.float32).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"loaded {MODEL}: {n_par / 1e6:.0f} M parameters, fp16 size ≈ {2 * n_par / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
    # single-image satellite scenes with ≥ 15 kurgan labels, smallest first
    scenes = []
    for reg in sorted(os.listdir(D)):
        tifs = glob.glob(f"{D}/{reg}/04_*/*.tif")
        labs = glob.glob(f"{D}/{reg}/*/*_SpOr_курганы_*.geojson")
        if len(tifs) != 1 or not labs:
            continue
        polys = []
        for f in labs:
            d = json.load(open(f)); n = (d.get("crs") or {}).get("properties", {}).get("name", "EPSG:3857")
            tr = Transformer.from_crs("EPSG:" + n.split(":")[-1], 3857, always_xy=True).transform
            polys += [stf(tr, shape(ft["geometry"])) for ft in d["features"] if ft["geometry"] and not shape(ft["geometry"]).is_empty]
        if len(polys) >= 15:
            scenes.append((os.path.getsize(tifs[0]), reg, tifs[0], polys))
    scenes = sorted(scenes)[:int(max_scenes)]
    X, B, y, sid = [], [], [], []
    n_chips = 0; t_inf = 0.0
    for k, (_, reg, tif, polys) in enumerate(scenes):
        a, T, crs, res, cands = scene_candidates(tif, polys)
        chips, meta = [], []
        for x, yy, z, rm, sign, lab in cands:
            c = chip(a, T, res, x, yy)
            if c is not None:
                chips.append(c); meta.append((np.log(z + 1e-6), rm, sign, lab))
        for i in range(0, len(chips), 32):
            batch = proc(images=chips[i:i + 32], return_tensors="pt")
            t1 = time.time()
            with torch.no_grad():
                h = model(**batch).last_hidden_state
            t_inf += time.time() - t1; n_chips += len(chips[i:i + 32])
            emb = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy()
            X.append(emb)
        for m in meta:
            B.append(m[:3]); y.append(m[3]); sid.append(k)
        print(f"  scene {k + 1}/{len(scenes)} {reg[:28]}: {len(chips)} chips, {int(sum(m[3] for m in meta))} in kurgans | "
              f"{t_inf / max(n_chips, 1):.2f} s/chip", flush=True)
    X = np.concatenate(X); B = np.array(B, float); y = np.array(y); sid = np.array(sid)
    os.makedirs(OUT, exist_ok=True); np.savez(f"{OUT}/emb.npz", X=X, B=B, y=y, sid=sid)

    def loso(F):
        p = np.zeros(len(y))
        for s in np.unique(sid):
            tr = sid != s
            if y[tr].sum() == 0:
                continue
            m = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(F[tr], y[tr])
            p[~tr] = m.predict_proba(F[~tr])[:, 1]
        per = [roc_auc_score(y[sid == s], p[sid == s]) for s in np.unique(sid) if 0 < y[sid == s].sum() < (sid == s).sum()]
        return roc_auc_score(y, p), float(np.mean(per))
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    for name, F in (("appearance only (LoG response, scale, polarity)", B),
                    ("DINOv3-SAT frozen embedding", Xs),
                    ("DINOv3-SAT + appearance", np.c_[Xs, B])):
        a_, b_ = loso(F)
        print(f"{name:48s} leave-scene-out AUC pooled {a_:.3f}  per-scene {b_:.3f}")
    print(f"CPU inference {t_inf / max(n_chips, 1):.2f} s per chip ({n_chips} chips); "
          f"public-size estimate: ~{(t_inf / max(n_chips, 1)) * 30 * 300 / 60:.0f} min on this CPU for 300 candidates × 30 regions")


if __name__ == "__main__":
    main(*sys.argv[1:2])
