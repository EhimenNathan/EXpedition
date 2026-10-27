"""Site-adaptive physical dome detector for LiDAR kurgans (CPU), evaluated leave-one-site-out.

Candidates: local maxima of the σ=5 m local relief model (DTM − Gaussian(DTM, σ)) on a 0.5 m grid.
Score: candidate relief divided by the site's own background roughness (robust std of the same channel), i.e. a
site-normalised SNR, times a radial-symmetry factor (share of 16 rays whose relief falls monotonically to the
mound's edge). The site gate is label-free: the density of strong candidates per km² (natural clutter) is computed at
inference from the DTM itself. Labels are used only for scoring and for choosing thresholds on the other sites.

  KOZ3_RAW=data python experiments/claude/li_dome_detector.py
"""
import glob, json, os, sys, warnings
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from shapely.ops import transform as stf
from pyproj import Transformer
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from koz3 import dtm  # noqa: E402

D = os.environ.get("KOZ3_RAW", "data"); OUT = os.environ.get("KOZ3_OUT", "work/li_dome"); R = 0.5
NPROC = os.cpu_count()


def crs_of(f):
    n = (json.load(open(f)).get("crs") or {}).get("properties", {}).get("name", "EPSG:3857")
    return "EPSG:4326" if "CRS84" in n else "EPSG:" + n.split(":")[-1]


def candidates(d, v):
    """→ rows (x_px, y_px, relief5, snr, symmetry, radius_m) and site statistics (label-free)."""
    L5 = d - ndi.gaussian_filter(d, 5 / R)
    L2 = d - ndi.gaussian_filter(d, 2 / R)
    sm = ndi.gaussian_filter(L5, 1.0)
    good = v & ndi.binary_erosion(v, iterations=int(10 / R))
    q = sm[good]
    rough = 1.4826 * np.median(np.abs(q - np.median(q))) if q.size else 1.0     # robust background roughness
    mx = (sm == ndi.maximum_filter(sm, size=int(6 / R))) & good & (sm > 2 * rough)
    ys, xs = np.nonzero(mx)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rows = []
    H, W = d.shape
    for y, x in zip(ys, xs):
        h0 = sm[y, x]; prof = []
        for a in ang:                                        # radial profiles out to 12 m
            rr = np.arange(1, int(12 / R)); py = (y - rr * np.sin(a)).astype(int); px = (x + rr * np.cos(a)).astype(int)
            ok = (py >= 0) & (py < H) & (px >= 0) & (px < W)
            prof.append(sm[py[ok], px[ok]] if ok.all() else None)
        prof = [p for p in prof if p is not None]
        if len(prof) < 12:
            continue
        P = np.array(prof)
        edge = np.array([np.argmax(p < 0.5 * h0) if (p < 0.5 * h0).any() else len(p) for p in P])   # half-height radius
        rad = np.median(edge) * R
        mono = np.mean([np.all(np.diff(p[:max(2, e)]) <= 0.02) for p, e in zip(P, edge)])
        rows.append((x, y, h0, h0 / rough, mono, rad, L2[y, x]))
    area_km2 = good.sum() * R * R / 1e6
    return np.array(rows, float).reshape(-1, 7), rough, area_km2


def site(reg):
    cache = f"{OUT}/{reg}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); return reg, z["C"], float(z["rough"]), float(z["area"]), list(z["polys"])
    las = glob.glob(f"{D}/{reg}/01_*/*.las")[0]
    utm = "EPSG:" + json.load(open(f"{D}/{reg}/UTM.json"))["crs"].split(":")[-1]
    d, v, (x0, y1, _) = dtm.build(las, res=R)
    C, rough, area = candidates(d, v)
    C[:, 0] = x0 + (C[:, 0] + 0.5) * R; C[:, 1] = y1 - (C[:, 1] + 0.5) * R
    polys = []
    for f in glob.glob(f"{D}/{reg}/*/*_Li_курганы_*.geojson"):
        tr = Transformer.from_crs(crs_of(f), utm, always_xy=True).transform
        polys += [stf(tr, shape(ft["geometry"])) for ft in json.load(open(f))["features"]
                  if ft["geometry"] and not shape(ft["geometry"]).is_empty]
    os.makedirs(OUT, exist_ok=True)
    np.savez(cache, C=C, rough=rough, area=area, polys=np.array(polys, dtype=object))
    return reg, C, rough, area, polys


def evaluate(C, polys, snr_t, sym_t, hull_m=30):
    """Greedy one-to-one, point inside label polygon; only candidates within hull_m of the labelled area
    (labels exist only near the site)."""
    from shapely.ops import unary_union
    if not polys:
        return 0, 0, 0
    hull = unary_union(polys).convex_hull.buffer(hull_m)
    keep = (C[:, 3] >= snr_t) & (C[:, 4] >= sym_t)
    K = C[keep]; K = K[np.argsort(-K[:, 3])]
    inside = np.array([hull.contains(Point(x, y)) for x, y in K[:, :2]], bool) if len(K) else np.zeros(0, bool)
    K = K[inside]
    tree = STRtree(polys); used = set(); tp = 0
    for x, y in K[:, :2]:
        p = Point(x, y)
        for j in tree.query(p):
            if j not in used and polys[j].contains(p):
                used.add(j); tp += 1; break
    return tp, len(K) - tp, len(polys) - tp


def main():
    from multiprocessing import Pool
    regs = sys.argv[1:] or sorted({p.split("/")[-3] for p in glob.glob(f"{D}/*/01_*/*.las")})
    with Pool(min(3, NPROC)) as p:
        S = {r: (C, rough, area, polys) for r, C, rough, area, polys in p.imap_unordered(site, regs)}
    grid = [(s, m) for s in (3, 4, 5, 6, 8, 10) for m in (0.0, 0.5, 0.7, 0.85)]
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fn + 0.5 * fp, 1)
    print(f"{'site':28s} {'n':>4s} {'rough m':>8s} {'strong cands/km²':>17s} | leave-site-out chosen (snr, sym) → TP FP FN F1")
    tot = np.zeros(3)
    for r in sorted(S):
        others = [o for o in S if o != r]
        best = max(grid, key=lambda g: f1(*np.sum([evaluate(S[o][0], S[o][3], *g) for o in others], 0)))
        tp, fp, fn = evaluate(S[r][0], S[r][3], *best); tot += (tp, fp, fn)
        C, rough, area, polys = S[r]
        dens = ((C[:, 3] >= 5).sum() / max(area, 1e-6)) if len(C) else 0
        print(f"{r[:28]:28s} {len(polys):4d} {rough:8.3f} {dens:17.0f} | {best} → {tp:4d} {fp:5d} {fn:4d}  {f1(tp, fp, fn):.3f}", flush=True)
    print(f"pooled leave-site-out: TP {int(tot[0])} FP {int(tot[1])} FN {int(tot[2])}  F1 {f1(*tot):.3f}")


if __name__ == "__main__":
    main()
