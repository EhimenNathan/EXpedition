"""Decode out-of-fold satellite maps (heads.py) into points and score them with the official metric, nested per class.

Decoding per scene (token grid, 8 m): peaks = 3×3 local maxima on valid tokens with sub-token quadratic refinement.
  kurgans        heat ≥ t_k; class = damaged if share ≥ 0.5 else intact; hedge: the other class too when its share ≥ τ
  architecture   heat ≥ t_a          fortifications  heat ≥ t_f
  hillforts      smoothed mask (σ 1 token): if max ≥ g, one point at its argmax (confidence = max)
Organiser rule (27 Oct): every source image is an independent reference layer, so duplicates are removed only WITHIN a
scene (same class family within 6 m, hillforts within 60 m: keep the best) and the ground truth is layered: each label
is copied into every scene whose footprint contains its representative point (as scripts/build_optical_layer_gt.py).
Per class, the setting is chosen on the other folds (official per-class F1 of pooled counts); the honest satellite score
pools the held-out folds' counts. Folds: koz3.sp_prep.site_folds (20 km clusters).
  python experiments/claude/sp_dense/evaluate.py [oof_dir=work/sp_dense/oof]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
K = ("kurgany_povrezhdennye", "kurgany_tselye")
MOD = os.environ.get("KOZ3_MOD", "SpOr")
D = os.path.join(ROOT, os.environ.get("SP_DENSE_OUT", "work/sp_dense"))
T_K = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
TAUS = (1.1, 0.45, 0.4, 0.35, 0.3)
T_A = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
T_F = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
G_H = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)


def peaks(h, valid, tmin):
    from scipy.ndimage import maximum_filter
    m = (h == maximum_filter(h, 3, mode="constant")) & valid & (h >= tmin)
    out = []
    for r, c in np.argwhere(m):
        dr = dc = 0.0
        if 0 < r < h.shape[0] - 1:
            a, b, e = h[r - 1, c], h[r, c], h[r + 1, c]; den = a - 2 * b + e
            dr = float(np.clip(0.5 * (a - e) / den, -0.5, 0.5)) if den < 0 else 0.0
        if 0 < c < h.shape[1] - 1:
            a, b, e = h[r, c - 1], h[r, c], h[r, c + 1]; den = a - 2 * b + e
            dc = float(np.clip(0.5 * (a - e) / den, -0.5, 0.5)) if den < 0 else 0.0
        out.append((r + dr, c + dc, float(h[r, c]), int(r), int(c)))
    return out


def decode_scene(p, s, tmin=0.05):
    """All candidates of one scene (thresholds applied later): DataFrame region, fam, cls, score, share, x, y (3857)."""
    from pyproj import Transformer
    from scipy.ndimage import gaussian_filter
    gh, gw = p.shape[1:]
    vh, vw = s["valid"]; valid = np.zeros((gh, gw), bool); valid[:int(np.ceil(vh)), :int(np.ceil(vw))] = True
    tr = Transformer.from_crs(s["crs"], 3857, always_xy=True)
    rows = []
    for fam, ch in (("k", 0), ("arkhitektury", 2), ("fortifikatsii", 3)):
        for r, c, v, ir, ic in peaks(p[ch].astype(np.float32), valid, tmin):
            X, Y = s["T"] * (c + 0.5, r + 0.5)
            rows.append((fam, v, float(p[1, ir, ic]) if fam == "k" else np.nan, X, Y))
    g = gaussian_filter(np.where(valid, p[4].astype(np.float32), 0), 1.0)
    ir, ic = np.unravel_index(int(np.argmax(g)), g.shape)
    X, Y = s["T"] * (ic + 0.5, ir + 0.5)
    rows.append(("gorodishcha", float(g.max()), np.nan, X, Y))
    D = pd.DataFrame(rows, columns=["fam", "score", "share", "x", "y"])
    if len(D):
        D["x"], D["y"] = tr.transform(D.x.values, D.y.values)
    D["region"] = s["region"]; D["tif"] = s["tif"]
    return D


def merge_region(D):
    """Within one scene (source layer): same family within 6 m (hillforts 60 m) → keep the highest score."""
    from scipy.spatial import cKDTree
    out = []
    for (reg, tif, fam), g in D.groupby(["region", "tif", "fam"]):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
        rad = (60.0 if fam == "gorodishcha" else 6.0) * k
        t = cKDTree(g[["x", "y"]].values); used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            used[t.query_ball_point(g.loc[i, ["x", "y"]].values, rad)] = True; out.append(g.loc[i])
    return pd.DataFrame(out).reset_index(drop=True)


def emit(C, t_k, tau, t_a, t_f, g_h):
    rows = []
    for r in C.itertuples():
        if r.fam == "k" and r.score >= t_k:
            prim = K[0] if r.share >= 0.5 else K[1]; other = K[1] if prim == K[0] else K[0]
            rows.append((r.region, prim, r.score, r.x, r.y))
            if min(r.share, 1 - r.share) >= tau:
                rows.append((r.region, other, r.score * 0.99, r.x, r.y))
        elif (r.fam == "arkhitektury" and r.score >= t_a) or (r.fam == "fortifikatsii" and r.score >= t_f) \
                or (r.fam == "gorodishcha" and r.score >= g_h):
            rows.append((r.region, r.fam, r.score, r.x, r.y))
    return pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])


def fc(df, r=1.0):
    from shapely.geometry import Point
    feats = []
    for row in df.itertuples():
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(row.y / 6378137.0)) - np.pi / 2)))
        feats.append({"type": "Feature", "geometry": Point(row.x, row.y).buffer(r * k, 12).__geo_interface__,
                      "properties": {"class_name": row.cls, "confidence": float(row.score)}})
    return {"type": "FeatureCollection", "features": feats}


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fn + 0.5 * fp, 1e-9) if tp + fn else 0.0


def layered_gt(L, scene_files):
    """GT per source layer: every label copied into each scene whose footprint covers its representative point."""
    import geopandas as gpd
    from shapely.geometry import box
    from heads import load_scene
    parts = []
    for p in scene_files:
        s = load_scene(p); g = L[L.region == s["region"]]
        if not len(g):
            continue
        gh, gw = s["f"].shape[:2]; vh, vw = s["valid"]
        (x0, y0), (x1, y1) = s["T"] * (0, 0), s["T"] * (vw, vh)
        fp = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        keep = g.to_crs(s["crs"]).geometry.representative_point().map(fp.covers).to_numpy()
        parts.append(g[keep].assign(tif=s["tif"]))
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=L.crs)


def candidates(oof_dir):
    from heads import load_scene
    parts = []
    for p in sorted(glob.glob(os.path.join(ROOT, oof_dir, "*.npz"))):
        s = load_scene(os.path.join(D, "scenes", os.path.basename(p)))
        parts.append(decode_scene(np.load(p)["p"], s))
    return merge_region(pd.concat(parts, ignore_index=True))


def main(oof_dir=None):
    oof_dir = oof_dir or os.path.relpath(os.path.join(D, "oof"), ROOT)
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.score import score, WEIGHTS
    from koz3.sp_prep import site_folds
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == MOD]; fo = site_folds(L)
    C = candidates(oof_dir)
    regions = sorted(set(C.region)); C["fold"] = C.region.map(fo); folds = sorted(C.fold.unique())
    LG = layered_gt(L[L.region.isin(regions)], [os.path.join(D, "scenes", os.path.basename(p))
                                                  for p in sorted(glob.glob(os.path.join(ROOT, oof_dir, "*.npz")))])
    print(f"layered GT: {len(LG)} objects from {int(L.region.isin(regions).sum())} labels")
    L = LG
    gt = {f: to_gt_geojson(L[L.region.map(fo) == f]) for f in folds}
    print(f"{len(regions)} regions, {len(C)} candidates; GT per class: {L.class_name.value_counts().to_dict()}", flush=True)
    # per-class counts per fold for each class-specific setting (classes are scored independently)
    grid = {"k": [(t, tau) for t in T_K for tau in TAUS], "arkhitektury": list(T_A), "fortifikatsii": list(T_F), "gorodishcha": list(G_H)}
    CNT = {}
    for fam, opts in grid.items():
        for o in opts:
            args = dict(t_k=9, tau=9, t_a=9, t_f=9, g_h=9)
            if fam == "k":
                args.update(t_k=o[0], tau=o[1])
            else:
                args[{"arkhitektury": "t_a", "fortifikatsii": "t_f", "gorodishcha": "g_h"}[fam]] = o
            for f in folds:
                _, res = score(fc(emit(C[C.fold == f], **args)), gt[f])
                CNT[(fam, o, f)] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
    fam_classes = {"k": K, "arkhitektury": ("arkhitektury",), "fortifikatsii": ("fortifikatsii",), "gorodishcha": ("gorodishcha",)}
    tot, chosen = {}, {}
    for fam, opts in grid.items():
        for f in folds:
            def val(o):
                s = 0.0
                for c in fam_classes[fam]:
                    tp, fp, fn = np.sum([CNT[(fam, o, g)][c] for g in folds if g != f], 0)
                    s += WEIGHTS[c] * f1(tp, fp, fn)
                return s
            o = max(opts, key=val); chosen[(fam, f)] = o
            for c in fam_classes[fam]:
                tot[c] = np.add(tot.get(c, (0, 0, 0)), CNT[(fam, o, f)][c])
    pres = [c for c in tot if tot[c][0] + tot[c][2]]
    honest = sum(WEIGHTS[c] * f1(*tot[c]) for c in pres) / sum(WEIGHTS[c] for c in pres)
    print(f"\nSATELLITE honest (nested per class): {honest:.4f}")
    for c in pres:
        tp, fp, fn = (int(v) for v in tot[c])
        print(f"  {c:24s} F1 {f1(tp, fp, fn):.3f}  tp {tp} fp {fp} fn {fn}  P {tp / max(tp + fp, 1):.2f} R {tp / max(tp + fn, 1):.2f}")
    print("chosen:", {k: v for k, v in chosen.items()})
    C.to_parquet(os.path.join(ROOT, oof_dir.rstrip("/") + "_cands.parquet"))


if __name__ == "__main__":
    main(*sys.argv[1:])
