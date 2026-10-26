"""Fast, exact re-implementation of the official KOZ-3 greedy matching (compute_metrics.F1Calculator._soft_match).

Metric koz3-1.5.0 (16 Oct): Polygon GT: hit iff cov = area(P∩G)/min(area) >= 0.5 AND the prediction's REPRESENTATIVE
point (shapely representative_point = GEOS point-on-surface, always inside the prediction) lies in G (boundary counts);
v1.4.1 used the centroid. The best GT is the one with the highest cov. Point GT: hit iff the prediction CENTROID is within
r (2 m) of the point (unchanged); the best GT is the nearest. Predictions are processed in descending confidence;
one-to-one. A polygon hit needs the representative point inside G, a point hit needs the centroid within r, so STRtree
queries on those points are exact prefilters. Tested equal to the official function (python koz3/official/fastmatch.py).
Coordinates must be metric (UTM) — the official script converts EPSG:3857 to the true UTM zone first."""
import math
import numpy as np
from shapely import STRtree
from shapely.geometry import Point, Polygon

COV = 0.5
R_TOL = 2.0
EPS = 1e-6


def _centroid_in(c, g):
    return g.contains(c) or g.boundary.distance(c) < 1e-10


def soft_match(preds, gts, r_tol=R_TOL):
    """preds: list of (Polygon, confidence); gts: list of Polygon or Point. Returns (tp, fp, fn, matched_pairs)."""
    if not preds or not gts:
        return 0, len(preds), len(gts), []
    pts = [i for i, g in enumerate(gts) if isinstance(g, Point)]
    pls = [i for i, g in enumerate(gts) if not isinstance(g, Point)]
    tree_pl = STRtree([gts[i] for i in pls]) if pls else None
    tree_pt = STRtree([gts[i].buffer(r_tol, 16) for i in pts]) if pts else None
    used = set(); tp = 0; pairs = []
    for p, _ in sorted(preds, key=lambda x: x[1], reverse=True):
        if not isinstance(p, Polygon):
            continue
        c = p.centroid; rp = p.representative_point(); best, bj = float("-inf"), -1
        if tree_pl is not None:
            for k in tree_pl.query(rp):
                j = pls[k]
                if j in used:
                    continue
                g = gts[j]
                if not _centroid_in(rp, g):  # v1.5.0: representative point, not centroid
                    continue
                m = min(p.area, g.area)
                cov = p.intersection(g).area / m if m > 0 else 0.0
                if cov >= COV and cov > best + 1e-9:
                    best, bj = cov, j
        if tree_pt is not None:
            for k in tree_pt.query(c):
                j = pts[k]
                if j in used:
                    continue
                d = math.hypot(c.x - gts[j].x, c.y - gts[j].y)
                if d <= r_tol and -d > best + 1e-9:
                    best, bj = -d, j
        if bj >= 0:
            tp += 1; used.add(bj); pairs.append((p, gts[bj]))
    return tp, len(preds) - tp, len(gts) - tp, pairs


def prf(tp, fp, fn):
    p = tp / (tp + 0.5 * fp + EPS); r = tp / (tp + fn + EPS)
    return p, r, 2 * p * r / (p + r + EPS)


def best_threshold(scores_by_unit, gts_by_unit, thresholds):
    """For a set of units (each: list of (geom, conf) and list of GT), return per-threshold pooled (tp, fp, fn)."""
    out = {}
    for t in thresholds:
        tp = fp = fn = 0
        for u in scores_by_unit:
            a, b, c, _ = soft_match([x for x in scores_by_unit[u] if x[1] >= t], gts_by_unit[u])
            tp += a; fp += b; fn += c
        out[t] = (tp, fp, fn)
    return out


if __name__ == "__main__":  # equivalence test against the official implementation
    import os, sys, random
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import compute_metrics as CM
    calc = CM.F1Calculator(input_crs_mode="utm")
    rnd = random.Random(0)
    for trial in range(400):
        gts = []
        for _ in range(rnd.randint(0, 12)):
            x, y = rnd.uniform(0, 50), rnd.uniform(0, 50)
            gts.append(Point(x, y) if rnd.random() < 0.4 else Point(x, y).buffer(rnd.uniform(0.5, 4), 8))
        preds = []
        for _ in range(rnd.randint(0, 15)):
            x, y, r = rnd.uniform(0, 50), rnd.uniform(0, 50), rnd.uniform(0.5, 5)
            p = Point(x, y).buffer(r, 16)
            if rnd.random() < 0.4:  # concave crescent: centroid can lie outside it (v1.4.1 vs v1.5.0 differ here)
                p = p.difference(Point(x + 0.4 * r, y).buffer(0.9 * r, 16))
                p = p if p.geom_type == "Polygon" else max(p.geoms, key=lambda g: g.area)
            preds.append((p, rnd.random()))
        a = soft_match(preds, gts)[:3]
        b = calc._soft_match(preds, [(g, 1.0) for g in gts])[:3]
        assert a == b, (trial, a, b)
    print(f"fastmatch == official ({CM._METRIC_CFG.METRIC_VERSION}) on 400 random trials incl. concave predictions")
