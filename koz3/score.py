"""Exact fast scorer: official koz3-1.5.0 parsing, UTM transform and pair test (imported from compute_metrics.py),
with STRtree candidate search instead of the all-pairs loop. Self-test: python -m koz3.score
"""
import os
import sys

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import compute_metrics as cm  # noqa: E402

CALC = cm.F1Calculator()
WEIGHTS = dict(cm.CLASS_WEIGHTS)


def _match(preds, gts):
    if not preds or not gts:
        return 0, len(preds), len(gts)
    polys = [(i, g) for i, (g, _) in enumerate(gts) if not isinstance(g, Point)]
    pts = [(i, g) for i, (g, _) in enumerate(gts) if isinstance(g, Point)]
    tp_tree = STRtree([g for _, g in polys]) if polys else None
    pt_tree = STRtree([g for _, g in pts]) if pts else None
    used, tp = set(), 0
    for pred, _ in sorted(preds, key=lambda x: x[1], reverse=True):
        if not isinstance(pred, Polygon):
            continue
        cand = []
        if tp_tree is not None:
            cand += [polys[k][0] for k in tp_tree.query(pred)]
        if pt_tree is not None:
            cand += [pts[k][0] for k in pt_tree.query(pred.centroid.buffer(CALC.r_tolerance + 0.01))]
        best, best_j = float("-inf"), -1
        for j in sorted(set(cand)):          # same order as the official loop → same tie-breaks
            if j in used:
                continue
            hit, s = CALC._match_quality(pred, gts[j][0])
            if hit and s > best + 1e-9:
                best, best_j = s, j
        if best_j >= 0:
            used.add(best_j); tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def score(pred_fc, gt_fc, classes=None):
    """Returns (final_score, {class: dict(f1,p,r,tp,fp,fn)}). classes restricts which classes count."""
    P = CALC._collect_by_class(pred_fc)
    G = CALC._collect_by_class(gt_fc, preserve_points=True)
    res, num, den = {}, 0.0, 0.0
    for c in (classes or CALC.target_classes):
        tp, fp, fn = _match(P.get(c, []), G.get(c, []))
        p = tp / (tp + 0.5 * fp + CALC.epsilon); r = tp / (tp + fn + CALC.epsilon)
        f1 = 2 * p * r / (p + r + CALC.epsilon)
        res[c] = dict(f1=f1, p=p, r=r, tp=tp, fp=fp, fn=fn)
        if len(G.get(c, [])) > 0:
            num += f1 * WEIGHTS[c]; den += WEIGHTS[c]
    return (num / den if den else 0.0), res


def _selftest(trials=60, seed=0):
    rng = np.random.default_rng(seed)
    x0, y0 = 4_200_000.0, 7_500_000.0
    classes = list(WEIGHTS)
    for t in range(trials):
        def feat(c, conf, pt=False):
            x, y = x0 + rng.uniform(0, 400), y0 + rng.uniform(0, 400)
            if pt:
                g = {"type": "Point", "coordinates": [x, y]}
            else:
                r = rng.uniform(1, 30); n = int(rng.integers(4, 12))
                ang = np.sort(rng.uniform(0, 2 * np.pi, n))
                rad = r * rng.uniform(0.5, 1.0, n)
                ring = [[x + a * np.cos(b), y + a * np.sin(b)] for a, b in zip(rad, ang)]
                g = {"type": "Polygon", "coordinates": [ring + [ring[0]]]}
            return {"type": "Feature", "geometry": g, "properties": {"class_name": c, "confidence": conf}}
        gt = {"type": "FeatureCollection", "features": [feat(c, 1.0, pt=(c == "finds_points")) for c in rng.choice(classes, 60)]}
        pr = {"type": "FeatureCollection", "features": [feat(c, float(rng.uniform())) for c in rng.choice(classes, 80)]}
        a = CALC.calculate_f1(pr, gt)["final_score"]
        b = score(pr, gt)[0]
        assert abs(a - b) < 1e-12, (t, a, b)
    print(f"fast scorer == official on {trials} random trials")


if __name__ == "__main__":
    _selftest()
