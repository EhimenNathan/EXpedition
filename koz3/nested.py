"""Nested (honest) selection of post-processing parameters on OOF predictions.

For every candidate setting p and fold f we store per-class (tp, fp, fn) on fold f only. The setting applied to fold f
is the one that maximises the official weighted score of the OTHER folds' pooled counts; the honest score pools the
held-out folds' counts under their own choices. Also returns the pooled-best ("oracle", an upper bound).
"""
import numpy as np

from koz3.score import CALC, WEIGHTS, score


def weighted(counts):
    """counts: {cls: (tp, fp, fn)} → official weighted F1 over classes present in GT."""
    num = den = 0.0
    for c, (tp, fp, fn) in counts.items():
        if tp + fn == 0:
            continue
        p = tp / (tp + 0.5 * fp + CALC.epsilon); r = tp / (tp + fn + CALC.epsilon)
        num += WEIGHTS[c] * 2 * p * r / (p + r + CALC.epsilon); den += WEIGHTS[c]
    return num / den if den else 0.0


def _add(a, b):
    out = dict(a)
    for c, v in b.items():
        out[c] = tuple(np.add(out.get(c, (0, 0, 0)), v))
    return out


def fold_counts(make_fc, settings, folds, gt_by_fold, classes=None):
    """make_fc(setting, fold) → prediction FeatureCollection for that fold. → {setting: {fold: {cls: (tp,fp,fn)}}}"""
    C = {}
    for s in settings:
        C[s] = {}
        for f in folds:
            _, res = score(make_fc(s, f), gt_by_fold[f], classes)
            C[s][f] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
    return C


def nested(C):
    """→ (honest score, {fold: chosen setting}, oracle score, oracle setting, honest per-class counts)."""
    settings = list(C); folds = list(next(iter(C.values())))
    pooled = {s: _add_all(C[s].values()) for s in settings}
    oracle = max(settings, key=lambda s: weighted(pooled[s]))
    chosen, tot = {}, {}
    for f in folds:
        best = max(settings, key=lambda s: weighted(_add_all(v for g, v in C[s].items() if g != f)))
        chosen[f] = best
        tot = _add(tot, C[best][f])
    return weighted(tot), chosen, weighted(pooled[oracle]), oracle, tot


def _add_all(it):
    tot = {}
    for v in it:
        tot = _add(tot, v)
    return tot
