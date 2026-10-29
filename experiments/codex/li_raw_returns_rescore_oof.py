"""Test raw LAS return attributes as independent LiDAR candidate evidence.

The deployed CNN only sees a terrain surface and derived relief channels.  The
cached 2 m rasters retain point density, multi-return share and intensity from
the original point cloud.  This script adds site-normalised local and annular
statistics to the existing candidate table, then compares geographic OOF AUC
for the deployed morphometric features with and without the raw attributes.

This is a screening gate.  An exact metric fusion is justified only when raw
attributes improve held-out ranking consistently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def safe_mean(a):
    a = np.asarray(a, np.float32)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def extract_raw(frame: pd.DataFrame) -> pd.DataFrame:
    from pyproj import Transformer
    from koz3.li_train import labels, site_items

    items = {it["region"]: it for it in site_items(labels())}
    rows = []
    for region, group in frame.groupby("region", sort=True):
        path = ROOT / "data/las_returns" / f"{region}.npz"
        if not path.exists() or region not in items:
            rows.extend({"raw_ok": 0} for _ in range(len(group)))
            continue
        z = np.load(path)
        n = z["n"].astype(np.float32)
        multi = z["multi"].astype(np.float32)
        inten = z["inten"].astype(np.float32)
        x0, ytop, res = map(float, z["origin"])
        transform = Transformer.from_crs(3857, items[region]["meta"]["crs"], always_xy=True)

        valid = n > 0
        # Vendor levels differ strongly.  Site-relative robust scaling keeps
        # only local contrast that can transfer to an unseen acquisition.
        def robust(a):
            v = a[valid & np.isfinite(a)]
            if not len(v):
                return np.nan, 1.0
            med = float(np.median(v))
            q = float(np.percentile(v, 75) - np.percentile(v, 25))
            return med, max(q, 1e-3)

        n_med, n_iqr = robust(np.log1p(n))
        m_med, m_iqr = robust(multi)
        i_med, i_iqr = robust(inten)
        h, w = n.shape
        for row in group.itertuples():
            X, Y = transform.transform(float(row.x), float(row.y))
            col = int(np.floor((X - x0) / res))
            rr = int(np.floor((ytop - Y) / res))
            out = {"raw_ok": int(0 <= rr < h and 0 <= col < w)}
            if not out["raw_ok"]:
                rows.append(out)
                continue
            stats = {}
            for radius_m in (2, 4, 8, 16):
                radius = max(1, int(round(radius_m / res)))
                ya, yb = max(0, rr - radius), min(h, rr + radius + 1)
                xa, xb = max(0, col - radius), min(w, col + radius + 1)
                yy, xx = np.ogrid[ya:yb, xa:xb]
                disk = (yy - rr) ** 2 + (xx - col) ** 2 <= radius ** 2
                ok = disk & valid[ya:yb, xa:xb]
                for name, array, med, iqr in (
                    ("dens", np.log1p(n), n_med, n_iqr),
                    ("multi", multi, m_med, m_iqr),
                    ("inten", inten, i_med, i_iqr),
                ):
                    value = safe_mean((array[ya:yb, xa:xb][ok] - med) / iqr)
                    stats[f"raw_{name}_r{radius_m}"] = value
            for name in ("dens", "multi", "inten"):
                stats[f"raw_{name}_core_ann"] = stats[f"raw_{name}_r4"] - stats[f"raw_{name}_r16"]
                stats[f"raw_{name}_mid_ann"] = stats[f"raw_{name}_r8"] - stats[f"raw_{name}_r16"]
            out.update(stats)
            rows.append(out)
    return pd.DataFrame(rows, index=frame.index)


def crossfit(frame, columns):
    import lightgbm as lgb
    out = np.full(len(frame), np.nan, np.float32)
    for fold in sorted(frame.fold.unique()):
        train = frame.fold != fold
        test = frame.fold == fold
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=3,
            random_state=0,
            verbose=-1,
            n_jobs=12,
        ).fit(frame.loc[train, columns], frame.loc[train, "lab"])
        out[test] = model.predict_proba(frame.loc[test, columns])[:, 1]
    return out


def main():
    from sklearn.metrics import roc_auc_score

    cache = ROOT / "work/li/rescore_cands.parquet"
    frame = pd.read_parquet(cache).reset_index(drop=True)
    raw_cache = ROOT / "work/li/rescore_raw_returns.parquet"
    if raw_cache.exists():
        raw = pd.read_parquet(raw_cache)
    else:
        raw = extract_raw(frame)
        raw.to_parquet(raw_cache, index=False)
    frame = pd.concat([frame, raw], axis=1)
    excluded = {"region", "fold", "cls", "x", "y", "conf", "lab"}
    base_columns = [c for c in frame.columns if c not in excluded and not c.startswith("raw_") and c != "raw_ok"]
    raw_columns = [c for c in frame.columns if c.startswith("raw_")]
    print(f"{len(frame)} candidates, {int(frame.lab.sum())} positives, raw coverage {frame.raw_ok.mean():.3f}", flush=True)
    frame["p_base"] = crossfit(frame, base_columns)
    frame["p_raw"] = crossfit(frame, raw_columns)
    frame["p_both"] = crossfit(frame, base_columns + raw_columns)
    for name in ("peak", "conf", "p_base", "p_raw", "p_both"):
        print(name, "pooled OOF AUC", round(roc_auc_score(frame.lab, frame[name]), 4), flush=True)
    for fold, group in frame.groupby("fold"):
        print(
            "fold", int(fold),
            "base", round(roc_auc_score(group.lab, group.p_base), 4),
            "raw", round(roc_auc_score(group.lab, group.p_raw), 4),
            "both", round(roc_auc_score(group.lab, group.p_both), 4),
            flush=True,
        )
    frame[["region", "fold", "cls", "x", "y", "lab", "p_base", "p_raw", "p_both"] + raw_columns].to_parquet(
        ROOT / "work/li/rescore_raw_returns_oof.parquet", index=False
    )


if __name__ == "__main__":
    main()
