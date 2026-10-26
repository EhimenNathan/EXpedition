#!/usr/bin/env python3
"""Strict LORO audit of map-internal normalization for MagGr finds.

The deployed detector uses absolute filter amplitudes. Those depend on map rendering,
instrument gain, interpolation, and soil background. This experiment adds only
deployment-computable, label-free within-map ranks / robust z scores, then evaluates
every held-out region with thresholds selected on other regions' OOF predictions.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import average_precision_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_import/code/work/geo/finds_v3"
sys.path.insert(0, str(ROOT / "kaggle_import/code"))
from koz3.official.fastmatch import soft_match, prf

DROP = {"x", "y", "tgt", "uid", "region", "render"}
THRESHOLDS = [0.01, .02, .03, .05, .075, .10, .15, .20, .25, .30, .40, .50, .60]


def add_selfnorm(df):
    """Add features computable independently inside each source map."""
    out = df.copy()
    cols = [c for c in out.columns if c not in DROP and np.issubdtype(out[c].dtype, np.number)]
    for uid, idx in out.groupby("uid", sort=False).groups.items():
        x = out.loc[idx, cols]
        # Percentile ranks are invariant to monotone contrast/gain changes.
        ranks = x.rank(method="average", pct=True)
        ranks.columns = [f"rank__{c}" for c in cols]
        out.loc[idx, ranks.columns] = ranks.to_numpy(np.float32)
        # Robust standardized values retain local amplitude while suppressing gain.
        med = x.median(axis=0)
        scale = (x.quantile(.75) - x.quantile(.25)).replace(0, 1.0)
        z = ((x - med) / scale).clip(-8, 8)
        z.columns = [f"rz__{c}" for c in cols]
        out.loc[idx, z.columns] = z.to_numpy(np.float32)
    return out


def model(seed=0):
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=.025, num_leaves=25,
        min_child_samples=30, subsample=.8, subsample_freq=1,
        colsample_bytree=.65, reg_lambda=3.0, random_state=seed,
        n_jobs=8, verbose=-1)


def exact_nested(o, gt, pcol):
    res = {}
    for uid, g in o.groupby("uid"):
        gts = [Point(x, y) for x, y in gt[uid]]
        for t in THRESHOLDS:
            s = g[g[pcol] >= t]
            pred = [(Point(x, y).buffer(.25, 8), p)
                    for x, y, p in zip(s.x, s.y, s[pcol])]
            res[(uid, t)] = np.asarray(soft_match(pred, gts)[:3], float)
    regions = sorted(o.region.unique())
    totals = np.zeros(3); rows = []
    for r in regions:
        other = [u for u in o.uid.unique() if not u.startswith(r + "|")]
        mine = [u for u in o.uid.unique() if u.startswith(r + "|")]
        t = max(THRESHOLDS, key=lambda q: prf(*np.sum([res[u, q] for u in other], axis=0))[2])
        c = np.sum([res[u, t] for u in mine], axis=0)
        totals += c
        rows.append(dict(region=r, threshold=t, tp=c[0], fp=c[1], fn=c[2], f1=prf(*c)[2]))
    p, r, f = prf(*totals)
    return dict(P=p, R=r, F1=f, tp=totals[0], fp=totals[1], fn=totals[2]), pd.DataFrame(rows)


def main():
    parts = [pd.read_parquet(p) for p in sorted((SRC / "units").glob("*__orig.parquet"))]
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.drop_duplicates(subset=["uid", "x", "y"], keep="first").reset_index(drop=True)
    data = add_selfnorm(raw)
    base = [c for c in raw.columns if c not in DROP]
    rank = [c for c in data.columns if c.startswith("rank__")]
    rz = [c for c in data.columns if c.startswith("rz__")]
    variants = {"raw": base, "rank": rank, "raw_rank": base + rank,
                "rank_rz": rank + rz, "raw_rank_rz": base + rank + rz}
    regs = sorted(data.region.unique())
    out = data[["uid", "region", "x", "y", "tgt"]].copy()
    for name, feats in variants.items():
        pred = np.full(len(data), np.nan)
        for region in regs:
            te = data.region.eq(region).to_numpy()
            tr = ~te
            pred[te] = model().fit(data.loc[tr, feats], data.loc[tr, "tgt"]).predict_proba(data.loc[te, feats])[:, 1]
        out[f"p_{name}"] = pred
        print(name, "AP", round(average_precision_score(data.tgt, pred), 5), flush=True)
    gt = {k: np.asarray(v) for k, v in json.load(open(SRC / "gt.json", encoding="utf-8")).items()}
    summary = []
    details = []
    for name in variants:
        s, d = exact_nested(out, gt, f"p_{name}")
        summary.append(dict(variant=name, **s))
        d.insert(0, "variant", name); details.append(d)
        print(name, {k: round(v, 5) for k, v in s.items()}, flush=True)
    pd.DataFrame(summary).to_csv(Path(__file__).with_suffix(".summary.csv"), index=False)
    pd.concat(details).to_csv(Path(__file__).with_suffix(".folds.csv"), index=False)
    out.to_parquet(Path(__file__).with_suffix(".oof.parquet"), index=False)


if __name__ == "__main__":
    main()
