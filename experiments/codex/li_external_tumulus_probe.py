#!/usr/bin/env python3
"""Prospective transfer probe for ObuObuHub/tumulus-lidar-detector.

The released MIT/CC-BY-4.0 detector was trained on Danish and Romanian mounds.
This script applies its unchanged proposal generator and four-channel CNN to
our cached DTMs, then evaluates candidate ranking site-disjoint by the existing
LiDAR folds.  It is a go/no-go screen before any exact deployed fusion.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = Path(os.environ.get("TUMULUS_EXT", "/home/nathan/Expedition Competition/work/external/tumulus-lidar-detector"))
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/claude"), str(EXT / "tools")]


def extract(region, item):
    from li_dome_features import load_dtm
    import tumul_scan as ts
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    d, valid, x0, ytop = load_dtm(region)
    d = np.where(valid, d, np.nan)
    candidates, _ = ts.scan(d, 0.5)
    polygons = [g for cls, g in item["objs"] if cls in (0, 1)]
    tree = STRtree(polygons) if polygons else None
    rows = []
    for c in candidates:
        # lib_tumul works on an effective 2 m grid. Pixel centres differ by at
        # most 1 m from the convention below, far inside ordinary mound labels.
        x = x0 + (c["x2"] + 0.5) * 2.0
        y = ytop - (c["y2"] + 0.5) * 2.0
        # li_train stores label polygons in the 0.5 m network pixel grid.
        p = Point((x - item["meta"]["x0"]) / item["meta"]["res"],
                  (item["meta"]["ytop"] - y) / item["meta"]["res"])
        hit = -1
        if tree is not None:
            hit = next((int(j) for j in tree.query(p) if polygons[int(j)].covers(p)), -1)
        cnn = float(c["cnn"]) if np.isfinite(c["cnn"]) else 0.0
        fuse = float(c["fuse"])
        rows.append(dict(region=region, fold=item["fold"], x=x, y=y, hit=hit,
                         cnn=cnn, fuse=fuse, noisy_or=1.0-(1.0-cnn)*(1.0-fuse),
                         z_mf=c["z_mf"], mahal=c["mahal"], closed=float(c["closed"]),
                         pfp=float(c.get("pfp", 0.0)), n_gt=len(polygons)))
    del d, valid, candidates
    gc.collect()
    return pd.DataFrame(rows)


def counts(frame, threshold, score, n_gt):
    q = frame[frame[score] >= threshold].sort_values(score, ascending=False)
    used = set()
    tp = 0
    for h in q.hit.astype(int):
        if h >= 0 and h not in used:
            used.add(h); tp += 1
    return np.array([tp, len(q)-tp, n_gt-tp], dtype=float)


def f1(c):
    tp, fp, fn = c
    return 2*tp / max(2*tp + fn + 0.5*fp, 1.0)


def evaluate(data, sites):
    folds = sorted({int(v["fold"]) for v in sites.values()})
    for score in ("cnn", "fuse", "noisy_or"):
        grid = np.unique(np.r_[np.linspace(0.01, 0.99, 99), data[score].quantile(np.linspace(.01,.99,99)).values])
        total = np.zeros(3); chosen = {}
        for fold in folds:
            train_sites = [r for r,v in sites.items() if v["fold"] != fold]
            test_sites = [r for r,v in sites.items() if v["fold"] == fold]
            def pooled(th, regs):
                return sum((counts(data[data.region==r], th, score, sites[r]["n_gt"]) for r in regs), np.zeros(3))
            best = max(grid, key=lambda t: f1(pooled(t, train_sites)))
            chosen[fold] = float(best)
            total += pooled(best, test_sites)
        oracle = max(grid, key=lambda t: f1(sum((counts(data[data.region==r],t,score,sites[r]["n_gt"]) for r in sites),np.zeros(3))))
        oc = sum((counts(data[data.region==r],oracle,score,sites[r]["n_gt"]) for r in sites),np.zeros(3))
        print(f"{score:8s} nested F1 {f1(total):.4f} TP/FP/FN {tuple(map(int,total))} | oracle {f1(oc):.4f} @ {oracle:.4f} {tuple(map(int,oc))} | chosen {chosen}")


def main():
    from koz3.li_train import labels, site_items
    L = labels(); items = {x["region"]: x for x in site_items(L)}
    out = ROOT / "work/li_external_tumulus_candidates.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    frames = [done] if len(done) else []
    have = set(done.region) if len(done) else set()
    for i, (region, item) in enumerate(sorted(items.items()), 1):
        if region in have:
            continue
        try:
            q = extract(region, item)
        except ValueError as exc:
            # Very small prepared rasters can be narrower than the largest
            # released matched-filter kernel. They contain no usable 80 m
            # inference window, so record an empty result and continue.
            print(f"[{i}/{len(items)}] {region}: skipped ({exc})", flush=True)
            q = pd.DataFrame(columns=["region", "fold", "x", "y", "hit", "cnn", "fuse", "noisy_or",
                                      "z_mf", "mahal", "closed", "pfp", "n_gt"])
        frames.append(q)
        allq = pd.concat(frames, ignore_index=True)
        allq.to_parquet(out, index=False)
        print(f"[{i}/{len(items)}] {region}: {len(q)} candidates, {int((q.hit>=0).sum())} in labels / {len([1 for c,g in item['objs'] if c in (0,1)])} GT", flush=True)
    data = pd.concat(frames, ignore_index=True)
    site_meta = {r: {"fold": it["fold"], "n_gt": len([1 for c,g in it["objs"] if c in (0,1)])} for r,it in items.items()}
    print(f"TOTAL {len(data)} candidates, {data.region.nunique()} sites, {(data.hit>=0).sum()} candidate-label incidences")
    evaluate(data, site_meta)


if __name__ == "__main__":
    main()
