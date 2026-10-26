"""Object-level evaluation that follows the organisers' stated rules (consultation #1, 2026-09-30):
- a prediction matches a GT object if they overlap OR the prediction's centroid falls inside the GT
  (point GT: centroid within R_tolerance metres);
- one GT <-> one prediction (greedy by score, then by overlap);
- per-class TP/FP/FN, with a false positive costing FP_W (about 0.5) of a miss;
- classes weighted by rarity.
The official formula arrives with the platform on 2026-10-12; `weighted_score` is a placeholder that exposes the
same levers (FP weight, class weights) so thresholds can be re-tuned the moment the script is available."""
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

FP_W = 0.5


def match(preds, gts, r_tol=None):
    """preds: list of (geom, score); gts: list of geom (Polygon or Point). Returns matched pairs, unmatched pred idx, unmatched gt idx."""
    if not preds:
        return [], [], list(range(len(gts)))
    if not gts:
        return [], list(range(len(preds))), []
    tree = STRtree(gts)
    cand = []
    for i, (pg, sc) in enumerate(preds):
        c = pg.centroid
        q = pg.buffer(r_tol) if r_tol else pg
        for j in tree.query(q):
            g = gts[j]
            if g.geom_type == "Point":
                ok = r_tol is not None and c.distance(g) <= r_tol
                ov = -c.distance(g)
            else:
                inter = pg.intersection(g).area if pg.area > 0 else 0.0
                ok = inter > 0 or g.contains(c)
                ov = inter / max(pg.union(g).area, 1e-9)
            if ok:
                cand.append((sc, ov, i, j))
    cand.sort(key=lambda t: (-t[0], -t[1]))
    up, ug, pairs = set(), set(), []
    for sc, ov, i, j in cand:
        if i in up or j in ug: continue
        up.add(i); ug.add(j); pairs.append((i, j, ov))
    return pairs, [i for i in range(len(preds)) if i not in up], [j for j in range(len(gts)) if j not in ug]


def fbeta(tp, fp, fn, beta):
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return (1 + beta ** 2) * p * r / max(beta ** 2 * p + r, 1e-9)


def counts_at(preds, gts, thr, r_tol=None):
    P = [(g, s) for g, s in preds if s >= thr]
    pairs, upred, ugt = match(P, gts, r_tol)
    return len(pairs), len(upred), len(ugt)


def ap_ar(preds, gts, r_tol=None, n=101):
    """AP: area under the precision-recall curve from a score sweep (101-point interpolation, COCO-style but with the
    competition's overlap-or-centre matching); AR: recall with every prediction kept."""
    if not gts:
        return np.nan, np.nan
    scores = sorted({s for _, s in preds}, reverse=True)
    if not scores:
        return 0.0, 0.0
    thr = np.unique(np.quantile(scores, np.linspace(0, 1, min(len(scores), 60))))
    P, R = [], []
    for t in thr:
        tp, fp, fn = counts_at(preds, gts, t, r_tol)
        P.append(tp / max(tp + fp, 1e-9)); R.append(tp / max(tp + fn, 1e-9))
    P, R = np.array(P), np.array(R); o = np.argsort(R)
    P, R = P[o], R[o]
    Pi = np.maximum.accumulate(P[::-1])[::-1]
    grid = np.linspace(0, 1, n)
    ap = np.mean([Pi[R >= g].max() if (R >= g).any() else 0.0 for g in grid])
    return float(ap), float(R.max())


def evaluate(pred_df, gt_df, classes, thresholds=None, class_weights=None, r_tol=None):
    """pred_df: columns [group, cls, geometry, score]; gt_df: [group, cls, geometry]. `group` = site|modality layer.
    Returns a per-class table plus a 'ALL' row, including the placeholder weighted score."""
    rows = []
    for c in classes:
        thr = (thresholds or {}).get(c, 0.5)
        tp = fp = fn = 0; allp, allg = [], []
        for grp in set(gt_df.group) | set(pred_df.group):
            P = pred_df[(pred_df.group == grp) & (pred_df.cls == c)]
            G = gt_df[(gt_df.group == grp) & (gt_df.cls == c)]
            preds = list(zip(P.geometry, P.score)); gts = list(G.geometry)
            a, b, d = counts_at(preds, gts, thr, r_tol); tp += a; fp += b; fn += d
            allp += [(g, s, grp) for g, s in preds]; allg += [(g, grp) for g in gts]
        # AP/AR pooled across groups: offset geometries are not needed because matching is done per group
        ap_num, ar_num, n_g = 0.0, 0.0, 0
        for grp in set(g for _, g in allg):
            pp = [(g, s) for g, s, gg in allp if gg == grp]; gg_ = [g for g, gg in allg if gg == grp]
            ap, ar = ap_ar(pp, gg_, r_tol)
            ap_num += ap * len(gg_); ar_num += ar * len(gg_); n_g += len(gg_)
        rows.append(dict(cls=c, n_gt=tp + fn, n_pred=tp + fp, TP=tp, FP=fp, FN=fn,
                         P=tp / max(tp + fp, 1e-9), R=tp / max(tp + fn, 1e-9),
                         F1=fbeta(tp, fp, fn, 1), F05=fbeta(tp, fp, fn, 0.5), F2=fbeta(tp, fp, fn, 2),
                         AP=ap_num / max(n_g, 1), AR=ar_num / max(n_g, 1),
                         Fw=tp / max(tp + FP_W * fp + fn, 1e-9)))
    T = pd.DataFrame(rows)
    present = T.n_gt > 0
    w = pd.Series(class_weights or {c: (1.0 / n if n > 0 else 0.0) for c, n in zip(T.cls, T.n_gt)})
    w = w.reindex(T.cls).fillna(0.0) * present.values
    w = w / max(w.sum(), 1e-12)
    allrow = dict(cls="ALL", n_gt=T.n_gt.sum(), n_pred=T.n_pred.sum(), TP=T.TP.sum(), FP=T.FP.sum(), FN=T.FN.sum(),
                  FP_absent_cls=int(T.loc[~present, "FP"].sum()))
    for k in ("P", "R", "F1", "F05", "F2", "AP", "AR", "Fw"):
        allrow[k] = float((T[k].values * w.values).sum())
    return pd.concat([T, pd.DataFrame([allrow])], ignore_index=True)


def pixel_iou(pred_masks, gt_masks, valid=None):
    """pred/gt: (C,H,W) bool. Returns per-class IoU and mIoU over classes present in GT or prediction."""
    ious = []
    for c in range(pred_masks.shape[0]):
        p, g = pred_masks[c], gt_masks[c]
        if valid is not None: p, g = p & valid, g & valid
        u = (p | g).sum()
        ious.append(np.nan if u == 0 else (p & g).sum() / u)
    return np.array(ious), float(np.nanmean(ious)) if np.isfinite(ious).any() else np.nan
