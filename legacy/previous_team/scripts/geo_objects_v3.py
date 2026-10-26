"""Geophysics objects (object_poly) v3: pixel LightGBM ensemble + rendering augmentation + palette channels +
hard negatives + honest per-method post-processing.

    python scripts/geo_objects_v3.py extract [--exclude 200_]   # stage A: sampled training pixels per unit & rendering
    python scripts/geo_objects_v3.py model   [--exclude 200_]   # stage B: leave-one-region-out probability maps + eval

Leakage control: leave-one-region-out; augmented renderings are training rows of their own region only; the
post-processing (smoothing sigma, threshold, minimum area) is chosen per method on the OTHER regions; no map-resolution
or region-identity feature.
Per unit (MagGr 0.2 m, EtGr 0.1 m, TerFl 0.3 m): physics stack at object scales (0.3-2.4 m, background 5/15 m) on the
grey value; on colour-coded maps also a decoded rainbow scalar with a reduced stack ('pal_*'); HSV chroma; method flags.
Samples per unit and rendering: 3000 object pixels, 4000 random background pixels, 2000 hard negatives in a 0.5-2 m
ring around objects. Renderings for training: orig, aug_blur, aug_contour, aug_gamma_lo (see geo_finds_v3).
Model: 3 LightGBM seeds (different column subsampling), probability mean.
Output: work/geo/objects_v3/{oof/*.npz, counts.csv, report.csv}"""
import argparse, glob, json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import rasterio.features as RF
from rasterio import features
from shapely.geometry import shape
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import geo_data as GD, geo_feats as GF
from koz3.official.fastmatch import soft_match, prf

ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["extract", "model"]); ap.add_argument("--root", default="data_geo")
ap.add_argument("--exclude", default=""); ap.add_argument("--renders", default="orig"); ap.add_argument("--tag", default="")
a = ap.parse_args()
OUT = "work/geo/objects_v3"; os.makedirs(f"{OUT}/units", exist_ok=True)
GSD = {"MagGr": 0.2, "EtGr": 0.1, "TerFl": 0.3}; METHODS = list(GSD)
SC = (0.3, 0.6, 1.2, 2.4); PAL_SC = (0.6, 2.4)
RENDERS = ["orig", "aug_blur", "aug_contour", "aug_gamma_lo"]
NPOS, NNEG, NHARD = 3000, 4000, 2000


def render(lum, valid, kind):
    if kind == "orig":
        return lum
    v = lum[valid]; lo, hi = np.percentile(v, [1, 99]) if v.size else (0, 255)
    x = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)
    x = {"aug_blur": lambda x: ndi.gaussian_filter(x, 1.5), "aug_contour": lambda x: ndi.gaussian_filter(np.round(x * 11) / 11, 1.0),
         "aug_gamma_lo": lambda x: x ** 0.6}[kind](x)
    return (x * 255).astype(np.float32)


def hsv(rgb, valid):
    import cv2
    h = cv2.cvtColor(np.clip(np.moveaxis(rgb, 0, -1), 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32); h[~valid] = 0
    hue = h[..., 0] * (2 * np.pi / 180.0); s = h[..., 1] / 255
    return {"hue_sin": np.sin(hue) * s, "hue_cos": np.cos(hue) * s, "sat": s}


def unit_stack(u, kind="orig"):
    d = GD.load_unit(u, GSD[u["method"]]); v = d["valid"]
    names, A = GF.stack(render(d["lum"], v, kind), v, d["gsd"], scales_m=SC, bg_m=(5.0, 15.0))
    pal, is_col = GF.palette_scalar(d["rgb"], v)
    extra = hsv(d["rgb"], v)
    if is_col:  # decoded colour scale: a reduced stack as extra channels (not a replacement)
        pn, PA = GF.stack(pal * 255, v, d["gsd"], scales_m=PAL_SC, bg_m=(5.0, 15.0))
        for n, ch in zip(pn, PA):
            if n in ("z", "hp5", "absz", "log0.6", "log2.4", "g2.4", "std1.5"):
                extra[f"pal_{n}"] = ch
    else:
        for n in ("z", "hp5", "absz", "log0.6", "log2.4", "g2.4", "std1.5"):
            extra[f"pal_{n}"] = np.zeros_like(d["lum"])
    for k_, ch in extra.items():
        names.append(k_); A = np.concatenate([A, ch[None].astype(np.float32)])
    const = {f"m_{m}": float(u["method"] == m) for m in METHODS}
    const.update(is_profile=float(u["kind"] == "profile"), is_colour=float(is_col))
    lab = np.zeros(v.shape, bool)
    if d["objects"]:
        lab = features.rasterize([(g, 1) for g in d["objects"]], out_shape=v.shape, transform=d["T"]).astype(bool)
    return d, names, A, const, lab


def to_frame(names, A, const, rr, cc):
    X = {n: A[i][rr, cc] for i, n in enumerate(names)}
    for k_, val in const.items():
        X[k_] = np.full(len(rr), val, np.float32)
    return pd.DataFrame(X)


def uid_of(u):
    return f"{u['region']}|{u['method']}|{u['key']}"


def units():
    U = [u for u in GD.discover(a.root) if u["objects"]]
    return [u for u in U if not (a.exclude and u["region"].startswith(tuple(a.exclude.split(","))))]


def extract():
    rng = np.random.default_rng(0)
    for u in units():
        for kind in RENDERS:
            p = f"{OUT}/units/{uid_of(u).replace('|', '__')}__{kind}.parquet"
            if os.path.exists(p):
                continue
            t0 = time.time()
            try:
                d, names, A, const, lab = unit_stack(u, kind)
            except Exception as e:
                print("FAILED", uid_of(u), kind, repr(e)[:160], flush=True); continue
            v = d["valid"]
            ring = ndi.binary_dilation(lab, iterations=max(1, int(2.0 / d["gsd"]))) & ~ndi.binary_dilation(lab, iterations=max(1, int(0.5 / d["gsd"])))
            pick = lambda m, n: (lambda ix: ix[rng.choice(len(ix), min(len(ix), n), replace=False)] if len(ix) else ix)(np.argwhere(m & v))
            pos, neg, hard = pick(lab, NPOS), pick(~lab, NNEG), pick(ring, NHARD)
            pts = np.r_[pos, neg, hard]
            X = to_frame(names, A, const, pts[:, 0], pts[:, 1]).astype(np.float32)
            X["tgt"] = np.r_[np.ones(len(pos)), np.zeros(len(neg) + len(hard))].astype(np.int8)
            X["uid"] = uid_of(u); X["region"] = u["region"]; X["method"] = u["method"]; X["render"] = kind
            X.to_parquet(p); del A
            print(f"{uid_of(u)} {kind}: {len(X)} px ({len(pos)} pos), {time.time() - t0:.0f}s", flush=True)


def polygons(prob, valid, T, gsd, sigma, thr, min_area):
    m = (ndi.gaussian_filter(prob, sigma) >= thr) & valid if sigma > 0 else (prob >= thr) & valid
    m = ndi.binary_opening(m); lab, n = ndi.label(m)
    if n == 0:
        return []
    means = ndi.mean(prob, lab, np.arange(1, n + 1)); sizes = ndi.sum(np.ones_like(prob), lab, np.arange(1, n + 1))
    out = []
    for geom, val in RF.shapes(lab.astype(np.int32), mask=lab > 0, transform=T):
        i = int(val) - 1
        if sizes[i] * gsd * gsd < min_area:
            continue
        g = shape(geom).buffer(0)
        g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area)
        out.append((g.simplify(gsd, preserve_topology=True), float(means[i])))
    return out


def model():
    import lightgbm as lgb
    # memory-lean load: float32 matrix preallocated, metadata kept as small arrays
    renders = RENDERS if a.renders == "all" else a.renders.split(",")
    files = [p for p in sorted(glob.glob(f"{OUT}/units/*.parquet")) if p.rsplit("__", 1)[-1][:-8] in renders]
    if a.exclude:
        files = [p for p in files if not os.path.basename(p).startswith(tuple(a.exclude.split(",")))]
    import pyarrow.parquet as pq
    meta_cols = ("tgt", "uid", "region", "method", "render")
    feats = [c for c in pq.read_schema(files[0]).names if c not in meta_cols]
    n = sum(pq.read_metadata(p).num_rows for p in files)
    XS = np.empty((n, len(feats)), np.float32); YS = np.empty(n, np.int8); RS = np.empty(n, object); i0 = 0
    for p in files:
        t = pd.read_parquet(p); k = len(t)
        XS[i0:i0 + k] = t[feats].to_numpy(np.float32); YS[i0:i0 + k] = t.tgt.to_numpy(); RS[i0:i0 + k] = t.region.to_numpy(); i0 += k
    print(f"{n} training pixels from {len(files)} unit-renderings ({renders}), {len(set(RS))} regions, {len(feats)} features", flush=True)
    U = {uid_of(u): u for u in units()}
    counts = []
    GRID = [(s, t, m) for s in (0.0, 1.0, 2.0) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for m in (0.3, 1.0, 3.0)]
    for r in sorted(set(RS)):
        tr = RS != r; Xtr = XS[tr]; ytr = YS[tr]
        models = [lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.7,
                                     subsample_freq=1, colsample_bytree=cs, reg_lambda=2.0, random_state=sd, n_jobs=4, verbose=-1).fit(Xtr, ytr)
                  for sd, cs in ((0, 0.7), (1, 0.5), (2, 0.9))]
        del Xtr
        for uid, u in U.items():
            if u["region"] != r:
                continue
            d, names, A, const, lab = unit_stack(u, "orig")
            v = d["valid"]; rr, cc = np.nonzero(v); prob = np.zeros(v.shape, np.float32)
            for c0 in range(0, len(rr), 150_000):
                sl = slice(c0, c0 + 150_000)
                Xp = to_frame(names, A, const, rr[sl], cc[sl])[feats].to_numpy(np.float32)
                prob[rr[sl], cc[sl]] = np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)
            del A
            os.makedirs(f"{OUT}/oof{a.tag}", exist_ok=True)
            np.savez_compressed(f"{OUT}/oof{a.tag}/{uid.replace('|', '__')}.npz", prob=prob.astype(np.float16), valid=v)
            for s_, t_, m_ in GRID:
                tp, fp, fn, _ = soft_match(polygons(prob, v, d["T"], d["gsd"], s_, t_, m_), d["objects"])
                counts.append(dict(uid=uid, region=r, method=u["method"], kind=u["kind"], sigma=s_, thr=t_, min_area=m_, tp=tp, fp=fp, fn=fn))
        print(f"region {r} done", flush=True)
        pd.DataFrame(counts).to_csv(f"{OUT}/counts{a.tag}.csv", index=False)
    R = pd.DataFrame(counts); rows = []
    key = ["sigma", "thr", "min_area"]
    for meth, g in R.groupby("method"):
        tot = np.zeros(3); chosen = {}
        for r in g.region.unique():
            o = g[g.region != r].groupby(key)[["tp", "fp", "fn"]].sum()
            if not len(o):
                continue
            best = max(o.index, key=lambda k: prf(*o.loc[k].values)[2]); chosen[r] = best
            tot += g[(g.region == r) & (g[key].apply(tuple, axis=1) == best)][["tp", "fp", "fn"]].sum().values
        allg = g.groupby(key)[["tp", "fp", "fn"]].sum(); ob = max(allg.index, key=lambda k: prf(*allg.loc[k].values)[2])
        rows.append(dict(method=meth, tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), F1_honest=round(prf(*tot)[2], 4),
                         deploy=ob, F1_oracle=round(prf(*allg.loc[ob].values)[2], 4)))
    allc = np.sum([[x["tp"], x["fp"], x["fn"]] for x in rows], axis=0)
    rows.append(dict(method="ALL object_poly (pooled)", tp=int(allc[0]), fp=int(allc[1]), fn=int(allc[2]), F1_honest=round(prf(*allc)[2], 4)))
    out = pd.DataFrame(rows); out.to_csv(f"{OUT}/report{a.tag}.csv", index=False); pd.set_option("display.width", 220); print(out.to_string(index=False))


if __name__ == "__main__":
    extract() if a.stage == "extract" else model()
