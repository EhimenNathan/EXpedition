"""Geometry-only mound classifier on CNN-UNCOVERED candidates (target population), honest over site folds; reuses
work/li_blob_cands.parquet (scripts/li_blob_gbm_oof.py). Excludes CNN heat / masks and site-identity features.
Reports hit rate vs volume on held-out folds, and within-site ranking (AUC per site)."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
C = pd.read_parquet("work/li_blob_cands.parquet"); U = C[~C.covered].copy()
drop = {"y", "x", "X", "Y", "lab", "region", "fold", "covered", "q", "heat", "heat_max5", "p_dam", "p_int", "site_dens", "site_r10_med"}
for name, feats in (("geometry+rank+context", [c for c in U.columns if c not in drop]),
                    ("geometry only (no ranks/neighbours)", [c for c in U.columns if c not in drop and not c.startswith(("rank_", "n25", "n100", "ns25", "ns100"))])):
    U["g"] = np.nan
    for f in sorted(U.fold.unique()):
        trn = U[U.fold != f]
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=5.0, random_state=0, verbose=-1, n_jobs=16, scale_pos_weight=5).fit(trn[feats], trn.lab)
        U.loc[U.fold == f, "g"] = m.predict_proba(U.loc[U.fold == f, feats])[:, 1]
    print(f"== {name}: OOF AUC {roc_auc_score(U.lab, U.g):.3f}  base rate {U.lab.mean():.4f}")
    for k in (100, 200, 400, 800, 1600):
        s = U.nlargest(k, "g"); print(f"  top {k}: hit rate {s.lab.mean():.3f}  hits {int(s.lab.sum())}  sites {s.region.nunique()}")
    a = [(r, roc_auc_score(g.lab, g.g), int(g.lab.sum())) for r, g in U.groupby("region") if 0 < g.lab.sum() < len(g) and g.lab.sum() >= 10]
    print("  per-site AUC (≥10 hits):", ", ".join(f"{r[:12]} {v:.2f}({n})" for r, v, n in a))
    U[f"g_{name[:4]}"] = U.g
U.to_parquet("work/li_blob_uncovered.parquet")
imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False); print("top:", ", ".join(imp.index[:10]))
