"""EtGr depth-stack consensus: CPU experiment with an independent pixel-GBM detector, leave-one-region-out.

Stage 1 (cached): per map image → pixel features → LightGBM trained on the other regions → probability map.
Stage 2: decode per layer (baseline) vs depth-stack consensus (support from adjacent depths + propagation of
confident neighbour objects into a layer where they were missed). Official metric (koz3.score), nested over regions.
"""
import glob, json, os, pickle, re, sys
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject, Resampling
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping
from shapely.ops import transform as stf
from pyproj import Transformer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from koz3.score import score
from koz3.nested import nested, weighted, _add_all

D = os.environ.get("KOZ3_RAW", "data"); OUT = os.environ.get("KOZ3_OUT", "work/etgr_stack")
RES = 0.05
os.makedirs(OUT, exist_ok=True)


def depth_key(path):
    m = re.search(r"карта_(\d+)", path); return int(m.group(1))


def load_layers():
    L = []
    for r in sorted(glob.glob(f"{D}/21[2-7]_*")):
        for t in sorted(glob.glob(f"{r}/*EtGr_карты/*.tif"), key=depth_key):
            stem = re.sub(r"^\d+_", "", os.path.basename(t)[:-4])
            lab = glob.glob(f"{r}/*/{stem}_ObjectPoly.geojson"); bnd = glob.glob(f"{r}/extra/{stem}_границы.geojson")
            L.append(dict(region=os.path.basename(r), tif=t, stem=stem, lab=lab[0], bnd=bnd[0] if bnd else None,
                          depth=depth_key(t)))
    return L


def file_crs(d):
    n = (d.get("crs") or {}).get("properties", {}).get("name", "EPSG:3857")
    return "EPSG:" + n.split(":")[-1]


def read_geoms(f, dst):
    d = json.load(open(f)); tr = Transformer.from_crs(file_crs(d), dst, always_xy=True).transform
    return [stf(tr, shape(x["geometry"])) for x in d["features"] if x["geometry"]]


def grid(lay):
    with rasterio.open(lay["tif"]) as ds:
        b = ds.bounds; crs = ds.crs
        W = int(np.ceil((b.right - b.left) / RES)); H = int(np.ceil((b.top - b.bottom) / RES))
        T = rasterio.transform.from_origin(b.left, b.top, RES, RES)
        a = np.zeros((3, H, W), np.float32)
        reproject(ds.read(indexes=[1, 2, 3]).astype(np.float32), a, src_transform=ds.transform, src_crs=crs,
                  dst_transform=T, dst_crs=crs, resampling=Resampling.average)
    geo = lambda f: read_geoms(f, crs)
    objs = geo(lay["lab"])
    inb = rasterize([(g, 1) for g in geo(lay["bnd"])], (H, W), transform=T) > 0 if lay["bnd"] else np.ones((H, W), bool)
    inb &= a.sum(0) > 0
    y = rasterize([(g, 1) for g in objs], (H, W), transform=T) > 0 if objs else np.zeros((H, W), bool)
    return a, inb, y, T, crs


def features(a, inb):
    from skimage.color import rgb2lab
    lab = rgb2lab(np.clip(a.transpose(1, 2, 0) / 255.0, 0, 1)).transpose(2, 0, 1)
    ch = list(a) + list(lab) + [np.hypot(lab[1], lab[2]), np.arctan2(lab[2], lab[1])]
    F = []
    for c in ch:
        v = c[inb]; mu, sd = v.mean(), v.std() + 1e-6
        z = np.where(inb, (c - mu) / sd, 0)
        pr = np.zeros_like(c); pr[inb] = (np.argsort(np.argsort(v)) / max(len(v) - 1, 1))
        F += [z, pr]
        for s in (2, 6, 20):                             # 0.1 / 0.3 / 1 m
            g = ndi.gaussian_filter(z, s); F += [g, z - g]
    return np.stack(F).astype(np.float32)


NPROC = os.cpu_count()


def _feat_job(lay):
    a, inb, y, T, crs = grid(lay)
    return (lay, features(a, inb), inb, y, T, crs)


def stage1():
    cache = f"{OUT}/probs.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import lightgbm as lgb
    from multiprocessing import Pool
    L = load_layers()
    with Pool(NPROC) as pool:
        data = pool.map(_feat_job, L)
    print("features done", len(data), "layers", flush=True)
    regions = sorted({d[0]["region"] for d in data}); rng = np.random.default_rng(0); probs = {}
    for r in regions:
        Xs, ys = [], []
        for lay, X, inb, y, _, _ in data:
            if lay["region"] == r:
                continue
            idx = np.flatnonzero(inb.ravel()); idx = rng.choice(idx, min(len(idx), 30000), replace=False)
            Xs.append(X.reshape(X.shape[0], -1)[:, idx].T); ys.append(y.ravel()[idx])
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=NPROC, verbose=-1).fit(np.concatenate(Xs), np.concatenate(ys))
        for lay, X, inb, y, T, crs in data:
            if lay["region"] == r:
                p = m.predict_proba(X.reshape(X.shape[0], -1).T)[:, 1].reshape(inb.shape)
                probs[lay["stem"]] = (lay, np.where(inb, p, 0).astype(np.float32), inb, T, crs)
        print("held out", r, flush=True)
    pickle.dump(probs, open(cache, "wb"))
    return probs


def components(p, inb, T, t, amin_m2):
    sm = ndi.gaussian_filter(p, 2)
    lbl, n = ndi.label((sm >= t) & inb)
    out = []
    for i, sl in enumerate(ndi.find_objects(lbl), 1):
        m = lbl[sl] == i
        if m.sum() * RES * RES < amin_m2:
            continue
        sc = float(sm[sl][m].mean())
        T2 = T * rasterio.Affine.translation(sl[1].start, sl[0].start)
        geoms = [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=T2) if v == 1]
        g = max(geoms, key=lambda q: q.area).simplify(RES)
        out.append([g, sc])
    return out


def to_fc(items, crs):
    tr = Transformer.from_crs(crs, 3857, always_xy=True).transform
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(stf(tr, g)), "properties": {"class_name": "object_poly", "confidence": s}}
        for g, s in items]}


def decode(probs, regs, st):
    t, amin, alpha, gamma = st
    out_by_region = {}
    for r in regs:
        lays = sorted([v for v in probs.values() if v[0]["region"] == r], key=lambda v: v[0]["depth"])
        comps = [components(p, inb, T, t, amin) for _, p, inb, T, _ in lays]
        hi = [components(p, inb, T, max(t, 0.5), amin) for _, p, inb, T, _ in lays] if gamma > 0 else None
        items = []
        for k, (lay, p, inb, T, crs) in enumerate(lays):
            nb = [j for j in (k - 1, k + 1) if 0 <= j < len(lays)]
            cur = []
            for g, s in comps[k]:
                sup = max([max([g.intersection(h).area / min(g.area, h.area) for h, _ in comps[j]] or [0]) for j in nb] or [0])
                cur.append([g, s * (1 + alpha * (sup - 0.5))])
            if gamma > 0:                                 # neighbour objects missing in this layer: copy them in
                cand = [(h, s) for j in nb for h, s in hi[j]]
                for h, s in cand:
                    if not any(h.intersection(g).area > 0.3 * min(h.area, g.area) for g, _ in cur):
                        cur.append([h, s * gamma])
            items += [(g, s) for g, s in cur]
        out_by_region[r] = (items, lays[0][4])
    return out_by_region


def gt_fc(regs):
    feats = []
    for r in regs:
        for f in glob.glob(f"{D}/{r}/*/*EtGr_карта*ObjectPoly.geojson"):
            for g in read_geoms(f, "EPSG:3857"):
                feats.append({"type": "Feature", "geometry": mapping(g), "properties": {"class_name": "object_poly"}})
    return {"type": "FeatureCollection", "features": feats}


_P = _GT = None


def _score_job(job):
    st, r = job
    items, crs = decode(_P, [r], st)[r]
    _, res = score(to_fc(items, crs), _GT[r], ["object_poly"])
    v = res["object_poly"]
    return (v["tp"], v["fp"], v["fn"])


def main():
    probs = stage1()
    regions = sorted({v[0]["region"] for v in probs.values()})
    gt = {r: gt_fc([r]) for r in regions}
    base = [(t, a, 0.0, 0.0) for t in (0.3, 0.4, 0.5, 0.6) for a in (0.05, 0.2)]
    treat = [(t, a, al, ga) for t in (0.3, 0.4, 0.5) for a in (0.05, 0.2) for al in (0.0, 0.5, 1.0) for ga in (0.0, 0.6, 0.9)
             if (al, ga) != (0.0, 0.0)]
    global _P, _GT
    _P, _GT = probs, gt
    from multiprocessing import Pool
    jobs = [(st, r) for st in base + treat for r in regions]
    with Pool(NPROC) as pool:
        res = pool.map(_score_job, jobs, chunksize=4)
    C = {}
    for (st, r), v in zip(jobs, res):
        C.setdefault(st, {})[r] = {"object_poly": v}
    for name, S in (("per-layer baseline", base), ("baseline + depth-stack consensus", base + treat)):
        h, ch, o, os_, tot = nested({s: C[s] for s in S})
        print(f"{name:34s} nested honest F1 {h:.4f}  (TP {tot['object_poly'][0]} FP {tot['object_poly'][1]} FN {tot['object_poly'][2]})"
              f"  oracle {o:.4f} @ {os_}  chosen {list(ch.values())}")
    hb, chb, *_ = nested({s: C[s] for s in base}); ht, cht, *_ = nested({s: C[s] for s in base + treat})
    for r in regions:
        print(f"   {r:26s} baseline {weighted(C[chb[r]][r]):.3f} → consensus {weighted(C[cht[r]][r]):.3f}  {cht[r]}")


if __name__ == "__main__":
    main()
