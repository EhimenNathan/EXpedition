"""Lightweight geophysics object detector (MagGr / EtGr / TerFl → object_poly), CPU only, leave-one-region-out.

Stage 1: per source image → resample to a method grid → per-image normalised colour/intensity + multi-scale contrast
and texture features → one pooled LightGBM (method one-hot) trained on the other regions → probability map.
Stage 2: per-image decoding (components of the smoothed probability; emitted as simplified polygons or as small
interior patches at the pole of inaccessibility), survey bounds and noise ("шум") masks applied, thresholds per
method chosen on the other regions (nested). Official metric (koz3.score), object_poly only.

  KOZ3_RAW=data python experiments/claude/geo_objects_cpu.py
"""
import glob, json, os, pickle, re, sys
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject, Resampling
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform as stf, polylabel
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = os.environ.get("KOZ3_RAW", "data"); OUT = os.environ.get("KOZ3_OUT", "work/geo_cpu")
RES = {"EtGr": 0.05, "MagGr": 0.15, "TerFl": 0.3}
METHODS = ("EtGr", "MagGr", "TerFl")
NPROC = os.cpu_count()


def file_crs(d):
    n = (d.get("crs") or {}).get("properties", {}).get("name", "EPSG:3857")
    return "EPSG:4326" if "CRS84" in n else "EPSG:" + n.split(":")[-1]


def read_geoms(f, dst):
    d = json.load(open(f)); tr = Transformer.from_crs(file_crs(d), dst, always_xy=True).transform
    out = []
    for x in d["features"]:
        if x["geometry"]:
            g = shape(x["geometry"])
            if not g.is_empty:
                out.append(stf(tr, g))
    return out


def layers():
    """One entry per source image with its own reference labels (public MagGr = grey maps only)."""
    L = []
    for r in sorted(glob.glob(f"{D}/2[0-1][0-9]_*")):
        reg = os.path.basename(r)
        for t in sorted(glob.glob(f"{r}/*/*.tif")):
            name = os.path.basename(t)[:-4]
            m = re.search(r"_(EtGr|MagGr|TerFl)", name)
            if not m:
                continue
            meth = m.group(1); stem = re.sub(r"^\d+_", "", name)
            if meth == "MagGr":
                if stem.endswith("_цвет"):                 # colour twin of the grey map: same labels, skip
                    continue
                key = stem[:-len("_серый")] if stem.endswith("_серый") else stem
            else:
                key = stem
            lab = glob.glob(f"{r}/*/{key}_ObjectPoly.geojson")
            if not lab:
                continue
            bnd = glob.glob(f"{r}/extra/{key}_границы.geojson") + glob.glob(f"{r}/*/{key}_границы.geojson")
            noi = glob.glob(f"{r}/extra/{key}_шум.geojson") + glob.glob(f"{r}/*/{key}_шум.geojson")
            L.append(dict(region=reg, method=meth, tif=t, key=key, lab=lab[0], bnd=bnd[0] if bnd else None,
                          noise=noi[0] if noi else None))
    return L


def grid(lay):
    res = RES[lay["method"]]
    with rasterio.open(lay["tif"]) as ds:
        b = ds.bounds; crs = ds.crs
        W = int(np.ceil((b.right - b.left) / res)); H = int(np.ceil((b.top - b.bottom) / res))
        T = rasterio.transform.from_origin(b.left, b.top, res, res)
        nb = min(3, ds.count)
        a = np.zeros((nb, H, W), np.float32)
        reproject(ds.read(indexes=list(range(1, nb + 1))).astype(np.float32), a, src_transform=ds.transform,
                  src_crs=crs, dst_transform=T, dst_crs=crs, resampling=Resampling.average)
        alpha = None
        if ds.count == 4:
            alpha = np.zeros((H, W), np.float32)
            reproject(ds.read(4).astype(np.float32), alpha, src_transform=ds.transform, src_crs=crs, dst_transform=T,
                      dst_crs=crs, resampling=Resampling.average)
    inb = np.ones((H, W), bool) if lay["bnd"] is None else \
        rasterize([(g, 1) for g in read_geoms(lay["bnd"], crs)], (H, W), transform=T) > 0
    inb &= a.sum(0) > 0
    if alpha is not None:
        inb &= alpha > 127
    noise = np.zeros((H, W), bool) if lay["noise"] is None else \
        rasterize([(g, 1) for g in read_geoms(lay["noise"], crs)], (H, W), transform=T) > 0
    objs = read_geoms(lay["lab"], crs)
    y = rasterize([(g, 1) for g in objs], (H, W), transform=T) > 0 if objs else np.zeros((H, W), bool)
    return a, inb, noise, y, T, crs, res


def features(a, inb, res, method):
    from skimage.color import rgb2lab
    if a.shape[0] == 3:
        lab = rgb2lab(np.clip(a.transpose(1, 2, 0) / 255.0, 0, 1)).transpose(2, 0, 1)
        ch = list(lab) + [np.hypot(lab[1], lab[2])]
    else:
        ch = [a[0]]
    F = []
    for c in ch:
        v = c[inb]
        if v.size == 0:
            v = np.zeros(1)
        lo, hi = np.percentile(v, [2, 98]); z = np.where(inb, (c - (lo + hi) / 2) / max(hi - lo, 1e-6), 0)
        F.append(z)
        for s_m in (0.1, 0.3, 1.0, 3.0):
            s = max(s_m / res, 0.5)
            g = ndi.gaussian_filter(z, s); F += [g, z - g]
            F.append(np.sqrt(np.maximum(ndi.gaussian_filter(z * z, s) - g * g, 0)))
    m = METHODS.index(method)
    F += [np.full(inb.shape, float(i == m), np.float32) for i in range(len(METHODS))]
    return np.stack(F).astype(np.float32)


def _feat(lay):
    a, inb, noise, y, T, crs, res = grid(lay)
    X = features(a, inb, res, lay["method"])
    os.makedirs(f"{OUT}/feat", exist_ok=True)
    f = f"{OUT}/feat/{lay['region']}__{lay['key']}.npy"
    np.save(f, X.astype(np.float16))
    rng = np.random.default_rng(0)
    pos = np.flatnonzero((y & inb).ravel()); neg = np.flatnonzero((~y & inb).ravel())
    take = np.r_[rng.choice(pos, min(len(pos), 6000), replace=False) if len(pos) else [],
                 rng.choice(neg, min(len(neg), 12000), replace=False) if len(neg) else []].astype(int)
    Xs = X.reshape(X.shape[0], -1)[:, take].T; ys = y.ravel()[take]
    return lay, f, inb, noise, T, crs, res, Xs, ys


def stage1():
    cache = f"{OUT}/probs.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import lightgbm as lgb
    from multiprocessing import Pool
    L = layers(); print(len(L), "labelled geophysics images:", {m: sum(l["method"] == m for l in L) for m in METHODS}, flush=True)
    with Pool(NPROC) as pool:
        data = pool.map(_feat, L)
    regions = sorted({d[0]["region"] for d in data}); probs = {}
    for r in regions:
        tr = [d for d in data if d[0]["region"] != r]
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=NPROC, verbose=-1)
        m.fit(np.concatenate([d[7] for d in tr]), np.concatenate([d[8] for d in tr]))
        for lay, f, inb, noise, T, crs, res, _, _ in data:
            if lay["region"] != r:
                continue
            X = np.load(f).astype(np.float32)
            p = m.predict_proba(X.reshape(X.shape[0], -1).T)[:, 1].reshape(inb.shape)
            probs[(lay["region"], lay["key"])] = (lay, np.where(inb & ~noise, p, 0).astype(np.float32), T, crs, res)
        print("held out", r, flush=True)
    pickle.dump(probs, open(cache, "wb"))
    return probs


def decode(p, T, crs, res, t, amin_m2, mode):
    sm = ndi.gaussian_filter(p, max(0.1 / res, 0.5))
    lbl, n = ndi.label(sm >= t)
    tr = Transformer.from_crs(crs, 3857, always_xy=True).transform
    out = []
    for i, sl in enumerate(ndi.find_objects(lbl), 1):
        m = lbl[sl] == i
        if m.sum() * res * res < amin_m2:
            continue
        sc = float(sm[sl][m].mean())
        T2 = T * rasterio.Affine.translation(sl[1].start, sl[0].start)
        geoms = [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=T2) if v == 1]
        g = max(geoms, key=lambda q: q.area)
        if mode == "patch":
            try:
                c = polylabel(g, res)
            except Exception:
                c = g.representative_point()
            g = Point(c.x, c.y).buffer(0.25, 16)
        else:
            g = g.simplify(res)
        out.append({"type": "Feature", "geometry": mapping(stf(tr, g)),
                    "properties": {"class_name": "object_poly", "confidence": sc}})
    return out


_P = _GT = None


def _job(job):
    from koz3.score import score
    st, r = job
    t_by, amin, mode = st
    feats = []
    for (reg, key), (lay, p, T, crs, res) in _P.items():
        if reg == r:
            feats += decode(p, T, crs, res, t_by[lay["method"]], amin, mode)
    _, res_ = score({"type": "FeatureCollection", "features": feats}, _GT[r], ["object_poly"])
    v = res_["object_poly"]
    return (v["tp"], v["fp"], v["fn"])


def main():
    global _P, _GT
    from multiprocessing import Pool
    from koz3.nested import nested, weighted, _add_all
    probs = stage1()
    regions = sorted({k[0] for k in probs})
    _GT = {}
    for r in regions:
        feats = []
        for (reg, key), (lay, *_rest) in probs.items():
            if reg == r:
                feats += [{"type": "Feature", "geometry": mapping(g), "properties": {"class_name": "object_poly"}}
                          for g in read_geoms(lay["lab"], "EPSG:3857")]
        _GT[r] = {"type": "FeatureCollection", "features": feats}
    _P = probs
    ts = (0.3, 0.4, 0.5, 0.6, 0.7)
    S = [({"EtGr": a, "MagGr": b, "TerFl": c}, am, md) for a in ts for b in ts for c in (0.4, 0.5, 0.6)
         for am in (0.05, 0.3) for md in ("poly", "patch")]
    key = lambda s: (tuple(sorted(s[0].items())), s[1], s[2])
    jobs = [(s, r) for s in S for r in regions]
    with Pool(NPROC) as pool:
        res = pool.map(_job, jobs, chunksize=8)
    C = {}
    for (s, r), v in zip(jobs, res):
        C.setdefault(key(s), {})[r] = {"object_poly": v}
    h, ch, o, os_, tot = nested(C)
    tp, fp, fn = tot["object_poly"]
    print(f"\nnested honest object_poly F1 {h:.4f}  TP {tp} FP {fp} FN {fn} (GT {tp + fn})  | oracle {o:.4f} @ {os_}")
    for r in regions:
        print(f"   {r:28s} chosen {ch[r]}  F1 {weighted(C[ch[r]][r]):.3f}  counts {C[ch[r]][r]['object_poly']}")


if __name__ == "__main__":
    main()
