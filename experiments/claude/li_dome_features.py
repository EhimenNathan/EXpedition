"""Experiment 2: feature-engineered dome candidates on all cached 0.5 m DTMs, ranker ablation leave-one-site-out.

Candidates: σ=5 m local-relief maxima (koz3-style relief) on the cached DTM (li_stream_dtm.py).
Feature families (cheap, physical; kept only if they survive leave-site-out validation):
  base     relief h (σ=5 m), site SNR, radial symmetry, half-height radius, fine relief (σ=2 m)   [li_dome_detector]
  msrm     multi-scale relief at σ = 1, 2, 3, 5, 8, 12, 20 m (Orengo & Petrie multi-scale relief model)
  ld       local dominance: mean elevation angle from a 2–10 m ring up to the candidate
  pit      robber-pit signature: rim maximum minus centre in the σ=2 m relief (looted, ring-shaped mounds)
  curv     Laplacian of the σ=2 m smoothed DTM at the centre
  site     per-site percentile rank and robust z-score of h, snr, rad, l2 (group-normalised: domain generalisation)
  nbr      candidate neighbourhood: counts within 20 / 50 m, 1st / 3rd nearest-neighbour distance, neighbours' mean h
Labels: candidate inside a LiDAR kurgan label; only candidates inside the labelled zone (convex hull of all LiDAR
labels of the site + 30 m) are used, since labels exist only near the site.

  KOZ3_RAW=data KOZ3_OUT=work/li_dome python experiments/claude/li_dome_features.py
"""
import glob, json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from li_dome_detector import candidates  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
D = os.environ.get("KOZ3_RAW", os.path.join(_ROOT, "data", "train-expds-3"))         # team layout (koz3.labels.DATA)
OUT = os.environ.get("KOZ3_OUT", os.path.join(_ROOT, "work", "li_dome")); R = 0.5
NPROC = os.cpu_count()
FAMILIES = {
    "base": ["h", "snr", "sym", "rad", "l2", "rough", "dens"],
    "msrm": [f"ms{s}" for s in (1, 2, 3, 5, 8, 12, 20)],
    "ld": ["ld"], "pit": ["pit"], "curv": ["curv"],
    "site": [f"{c}_{k}" for c in ("h", "snr", "rad", "l2") for k in ("rk", "z")],
    "nbr": ["n20", "n50", "nn1", "nn3", "nbr_h"],
    "symcv": ["cv5", "cv10"],
    "pcurv": ["plan_c", "prof_c"],
}


def load_dtm(reg):
    """0.5 m DTM: our stream cache (li_stream_dtm.py) or the team's 0.25 m prep (data/las_prep), block-averaged."""
    f = f"{OUT}/dtm/{reg}.npz"
    if os.path.exists(f):
        z = np.load(f); return z["d"], z["v"], *z["origin"][:2]
    from koz3 import dtm as dtm_mod
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    d, v, (x0, y1, res), _ = dtm_mod.load(os.path.join(root, "data", "las_prep", reg + ".npz"))
    k = int(round(R / res))
    if k > 1:
        H, W = (d.shape[0] // k) * k, (d.shape[1] // k) * k
        d = d[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3)); v = v[:H, :W].reshape(H // k, k, W // k, k).all((1, 3))
    return d.astype(np.float32), v, x0, y1


def crs_of(f):
    n = (json.load(open(f)).get("crs") or {}).get("properties", {}).get("name", "EPSG:3857")
    return "EPSG:4326" if "CRS84" in n else "EPSG:" + n.split(":")[-1]


def site(reg):
    cache = f"{OUT}/feat/{reg}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    from shapely.geometry import shape, Point
    from shapely.ops import transform as stf, unary_union
    from shapely.strtree import STRtree
    from pyproj import Transformer
    d, v, x0, y1 = load_dtm(reg)
    utm = "EPSG:" + json.load(open(f"{D}/{reg}/UTM.json"))["crs"].split(":")[-1]
    kur, allp = [], []
    for f in glob.glob(f"{D}/{reg}/*/*_Li_*.geojson"):
        tr = Transformer.from_crs(crs_of(f), utm, always_xy=True).transform
        g = [stf(tr, shape(ft["geometry"])) for ft in json.load(open(f))["features"]
             if ft["geometry"] and not shape(ft["geometry"]).is_empty]
        allp += g
        if "курган" in f:
            kur += g
    if not allp:
        return pd.DataFrame()
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    zone = unary_union(allp).convex_hull.buffer(30)
    zmask = rasterize([(zone.buffer(20), 1)], d.shape, transform=from_origin(x0, y1, R, R)) > 0
    C, rough, area = candidates(d, v & zmask)          # only the labelled zone (+ margin for the 10 m erosion)
    if not len(C):
        return pd.DataFrame()
    df = pd.DataFrame(C, columns=["px", "py", "h", "snr", "sym", "rad", "l2"])
    df["rough"] = rough; df["dens"] = (df.snr >= 5).sum() / max(area, 1e-6)
    py, px = df.py.astype(int).values, df.px.astype(int).values
    for s in (1, 2, 3, 5, 8, 12, 20):
        L = d - ndi.gaussian_filter(d, s / R); df[f"ms{s}"] = L[py, px]
    L2 = ndi.gaussian_filter(d - ndi.gaussian_filter(d, 2 / R), 0.5)
    ring_max = ndi.maximum_filter(L2, footprint=_ring(1.0, 4.0))
    df["pit"] = ring_max[py, px] - L2[py, px]
    ds = ndi.gaussian_filter(d, 2 / R)
    df["curv"] = ndi.laplace(ds)[py, px]
    zy, zx = np.gradient(ds, R); zyy, zyx = np.gradient(zy, R); zxy, zxx = np.gradient(zx, R)
    g2 = zx ** 2 + zy ** 2 + 1e-9
    prof = -(zxx * zx ** 2 + 2 * zxy * zx * zy + zyy * zy ** 2) / (g2 * np.sqrt(1 + g2) ** 3)   # profile curvature
    plan = -(zxx * zy ** 2 - 2 * zxy * zx * zy + zyy * zx ** 2) / (g2 ** 1.5)                   # plan curvature
    win = int(3 / R) * 2 + 1                                                                    # mean within ~3 m
    df["prof_c"] = ndi.uniform_filter(np.clip(prof, -5, 5), win)[py, px]
    df["plan_c"] = ndi.uniform_filter(np.clip(plan, -50, 50), win)[py, px]
    L5 = ndi.gaussian_filter(d - ndi.gaussian_filter(d, 5 / R), 1.0); a16 = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for rm, nm in ((5.0, "cv5"), (10.0, "cv10")):                                               # radial symmetry CV
        yy = np.clip((py[:, None] - rm / R * np.sin(a16)).astype(int), 0, d.shape[0] - 1)
        xx = np.clip((px[:, None] + rm / R * np.cos(a16)).astype(int), 0, d.shape[1] - 1)
        drop = L5[py, px][:, None] - L5[yy, xx]
        df[nm] = drop.std(1) / (np.abs(drop.mean(1)) + 1e-3)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False); ld = np.zeros(len(df))
    H, W = d.shape
    for k, (y, x) in enumerate(zip(py, px)):
        vals = []
        for rr in (4, 8, 12, 16, 20):                        # 2–10 m ring at 0.5 m
            yy = np.clip((y - rr * np.sin(ang)).astype(int), 0, H - 1); xx = np.clip((x + rr * np.cos(ang)).astype(int), 0, W - 1)
            vals.append(np.degrees(np.arctan((d[y, x] - d[yy, xx]) / (rr * R))))
        ld[k] = np.mean(vals)
    df["ld"] = ld
    for c in ("h", "snr", "rad", "l2"):
        x = df[c].values; med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med)) + 1e-9
        df[f"{c}_rk"] = pd.Series(x).rank(pct=True).values; df[f"{c}_z"] = (x - med) / mad
    df["x"] = x0 + (df.px + 0.5) * R; df["y"] = y1 - (df.py + 0.5) * R
    strong = df[df.snr >= 3]
    t = cKDTree(strong[["x", "y"]].values) if len(strong) > 3 else None
    XY = df[["x", "y"]].values
    if t is not None:
        df["n20"] = [len(q) - 1 for q in t.query_ball_point(XY, 20)]
        df["n50"] = [len(q) - 1 for q in t.query_ball_point(XY, 50)]
        dd, ii = t.query(XY, k=min(4, len(strong)))
        df["nn1"] = dd[:, 1]; df["nn3"] = dd[:, min(3, dd.shape[1] - 1)]
        hs = strong.h.values; df["nbr_h"] = [hs[i[1:]].mean() for i in ii]
    else:
        df[["n20", "n50", "nn1", "nn3", "nbr_h"]] = 0.0
    inz = np.array([zone.contains(Point(a, b)) for a, b in XY])
    df = df[inz].reset_index(drop=True)
    tree = STRtree(kur) if kur else None
    hit, depth = [], []
    for a, b in df[["x", "y"]].values:
        p = Point(a, b); j = -1
        if tree is not None:
            j = next((int(q) for q in tree.query(p) if kur[q].contains(p)), -1)
        hit.append(j); depth.append(kur[j].boundary.distance(p) if j >= 0 else 0.0)
    df["hit"] = hit; df["depth"] = depth; df["region"] = reg; df["n_kur"] = len(kur)
    os.makedirs(f"{OUT}/feat", exist_ok=True); df.to_parquet(cache)
    return df


def _ring(r0, r1):
    a, b = int(r0 / R), int(np.ceil(r1 / R)); yy, xx = np.mgrid[-b:b + 1, -b:b + 1]; rr = np.hypot(yy, xx)
    return (rr >= a) & (rr <= b)


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fn + 0.5 * fp, 1)


def site_eval(G, t, n_kur):
    K = G[G.p >= t].sort_values("p", ascending=False); used = set(); tp = 0
    for h in K.hit:
        if h >= 0 and h not in used:
            used.add(h); tp += 1
    return np.array([tp, len(K) - tp, n_kur - tp])


def ablation(D_, cols, name, hardneg=False):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    D_ = D_.copy(); D_["p"] = 0.0
    best = D_[D_.hit >= 0].sort_values("depth", ascending=False).groupby(["region", "hit"]).head(1).index
    D_["y_"] = 0; D_.loc[best, "y_"] = 1
    D_["train"] = (D_.hit < 0) | D_.index.isin(best)            # other candidates inside a mound: ambiguous, not trained on
    if hardneg:
        D_["train"] &= (D_.y_ == 1) | (D_.snr >= 3)               # negatives: only mound-like look-alikes
    for r in D_.region.unique():
        tr = (D_.region != r) & D_.train
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=NPROC).fit(D_.loc[tr, cols], D_.loc[tr, "y_"])
        te = D_.region == r
        D_.loc[te, "p"] = m.predict_proba(D_.loc[te, cols])[:, 1]
    nk = D_.groupby("region").n_kur.first(); regs = [r for r in nk.index if nk[r] > 0]
    ts = np.arange(0.1, 0.95, 0.05); tot = np.zeros(3); per = {}
    for r in regs:
        o = [q for q in regs if q != r]
        best = max(ts, key=lambda t: f1(*sum(site_eval(D_[D_.region == q], t, nk[q]) for q in o)))
        c = site_eval(D_[D_.region == r], best, nk[r]); tot += c; per[r] = f1(*c)
    E = D_[D_.train]; auc = roc_auc_score(E.y_, E.p)
    sa = [roc_auc_score(g.y_, g.p) for _, g in E.groupby("region") if 0 < g.y_.sum() < len(g)]
    print(f"{name:34s} AUC pooled {auc:.3f} per-site {np.mean(sa):.3f}  pooled F1 {f1(*tot):.3f}  TP {int(tot[0])} FP {int(tot[1])} FN {int(tot[2])}", flush=True)
    return D_, per


def main():
    from multiprocessing import Pool
    regs = sorted({os.path.basename(f)[:-4] for f in glob.glob(f"{OUT}/dtm/*.npz") + glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "las_prep", "*.npz"))})
    with Pool(min(3, NPROC)) as p:
        parts = p.map(site, regs)
    D_ = pd.concat([q for q in parts if len(q)], ignore_index=True)
    print(f"{D_.region.nunique()} sites with labels, {len(D_)} candidates in labelled zones, "
          f"{int((D_.hit >= 0).sum())} inside kurgan labels", flush=True)
    base = FAMILIES["base"]
    _, per0 = ablation(D_, base, "base (as experiment 1)")
    for fam in ("msrm", "ld", "pit", "curv", "site", "nbr"):
        ablation(D_, base + FAMILIES[fam], f"base + {fam}")
    for fam in ("symcv", "pcurv"):
        ablation(D_, base + FAMILIES[fam], f"base + {fam}")
    allc = sum(FAMILIES.values(), [])
    Dall, per1 = ablation(D_, allc, "all families")
    ablation(D_, allc, "all families + hard negatives", hardneg=True)
    print("\nper-site F1 base → all (sites with ≥ 20 kurgans):")
    for r in sorted(per1):
        if D_[D_.region == r].n_kur.iloc[0] >= 20:
            print(f"   {r[:30]:30s} {per0.get(r, 0):.3f} → {per1[r]:.3f}")
    Dall.to_parquet(f"{OUT}/ranked_all.parquet")


if __name__ == "__main__":
    main()
