"""Deployable geophysics finds model (previous team's v3 'base' recipe, honest LORO F1 0.7187) for our inference.

1. Train one LightGBM on every region's original-rendering candidates (cached by geo_finds_v3.py extract).
2. Threshold = best pooled over the leave-one-region-out OOF probabilities (oof.parquet, p_base).
3. Export booster JSON → work/geo/finds_deploy/{finds_gbm.json, finds.json}; check koz3.gbm_np == LightGBM.
4. Check our ported candidate extraction (koz3.geo_finds_core, scipy peaks) reproduces the cached features.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "kaggle_import/code/work/geo/finds_v3")
OUT = os.path.join(ROOT, "work/geo/finds_deploy")


def main():
    import lightgbm as lgb
    from shapely.geometry import Point
    from koz3.gbm_np import NumpyGBM
    from koz3.score import CALC
    os.makedirs(OUT, exist_ok=True)
    F = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{SRC}/units/*__orig.parquet"))], ignore_index=True)
    feats = [c for c in F.columns if c not in ("x", "y", "tgt", "uid", "region", "render")]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, random_state=0, n_jobs=8, verbose=-1).fit(F[feats], F.tgt)
    dump = m.booster_.dump_model()
    json.dump(dump, open(f"{OUT}/finds_gbm.json", "w"))
    sample = F[feats].sample(min(20000, len(F)), random_state=0)
    d = np.abs(NumpyGBM(dump).predict_proba(sample.values)[:, 1] - m.predict_proba(sample)[:, 1]).max()
    print(f"numpy GBM vs LightGBM max |Δp| = {d:.2e}")
    assert d < 1e-6
    # threshold from LORO OOF (pooled), with the official point rule (centroid within 2 m)
    O = pd.read_parquet(f"{SRC}/oof.parquet")
    GT = {k: np.array(v) for k, v in json.load(open(f"{SRC}/gt.json", encoding="utf-8")).items()}
    best = None
    for t in (0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        tp = fp = fn = 0
        for uid, g in O.groupby("uid"):
            s = g[g.p_base >= t]
            from koz3.score import _match
            a, b, c = _match([(Point(x, y).buffer(0.25, 8), p) for x, y, p in zip(s.x, s.y, s.p_base)],
                             [(Point(x, y), 1.0) for x, y in GT[uid]])
            tp += a; fp += b; fn += c
        f1 = 2 * tp / max(2 * tp + fn + 0.5 * fp, 1)
        print(f"thr {t:.3f} F1 {f1:.4f} (tp {tp} fp {fp} fn {fn})")
        best = max(best or (f1, t), (f1, t))
    json.dump(dict(features=feats, threshold=best[1], pred_radius_m=0.25, gsd=0.1, oof_f1=round(best[0], 4),
                   regions=sorted(F.region.unique()), rows=len(F)), open(f"{OUT}/finds.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("deploy threshold", best)
    # ported extraction == cached features (one small unit)
    from koz3 import geo_data, geo_finds_core
    units = geo_data.discover(os.path.join(ROOT, "kaggle_import/code/data_geo"))
    u = next(u for u in units if u["region"].startswith("204") and u["finds"])
    df = geo_finds_core.candidates(geo_data.load_unit(u, 0.1))
    ref = pd.read_parquet(glob.glob(f"{SRC}/units/204_*__orig.parquet")[0])
    common = [c for c in feats if c in df.columns]
    a = df.sort_values(["x", "y"]).reset_index(drop=True); b = ref.sort_values(["x", "y"]).reset_index(drop=True)
    print(f"ported extraction: {len(a)} vs cached {len(b)} candidates; max |Δfeature| "
          f"{np.nanmax(np.abs(a[common].values - b[common].values)) if len(a) == len(b) else 'n/a'}")


if __name__ == "__main__":
    main()
