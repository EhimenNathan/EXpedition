"""Fold-safe probe of cached DINOv2 patch tokens for satellite objectness.

This uses pre-existing frozen DINOv2 features from small satellite scenes; it
does not train a vision backbone. Fold 3 is evaluated only after the fixed
PCA+logistic pipeline is fit on folds 0-2.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.scene_cls import scenes
from sp_gor_localise_oof import patch_geo

def main():
    F = pd.read_parquet(Path(__file__).with_name("sp_morph_features.parquet"))
    T = np.load(ROOT / "work/spor/dinov2b_patch_tokens.npy", mmap_mode="r")
    chips = scenes()
    assert len(chips) == len(T)
    groups = {}
    for i, c in enumerate(chips):
        x, y, ins = patch_geo(c)
        xy = np.column_stack([x[ins], y[ins]])
        tok = np.asarray(T[i][ins.ravel()], dtype=np.float32)
        tok -= tok.mean(axis=0)
        if len(xy) > 2:
            spacing = float(np.median(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))))
        else:
            spacing = 50.0
        groups.setdefault(c["region"], []).append((cKDTree(xy), tok, spacing))
    X = np.full((len(F), T.shape[-1]), np.nan, np.float32)
    for region, idx in F.groupby("region").groups.items():
        if region not in groups:
            continue
        xy = F.loc[idx, ["x", "y"]].to_numpy()
        best = np.full(len(idx), np.inf)
        token = np.zeros((len(idx), T.shape[-1]), np.float32)
        for tree, tok, spacing in groups[region]:
            dist, pos = tree.query(xy)
            norm = dist / max(spacing, 1)
            use = (norm < 1.0) & (norm < best)
            token[use] = tok[pos[use]]
            best[use] = norm[use]
        m = best < np.inf
        X[np.asarray(idx)[m]] = token[m]
    mask = np.isfinite(X[:, 0])
    print("candidate tokens", int(mask.sum()), "of", len(F), "positive", int(F.loc[mask, "y_obj"].sum()),
          "folds", F.loc[mask].fold.value_counts().to_dict(), flush=True)
    F = F[mask].reset_index(drop=True); X = X[mask]
    y = F.y_obj.to_numpy()
    z = np.empty(len(F), np.float32)
    zb = np.empty(len(F), np.float32)
    for fold in (0, 1, 2, 3):
        tr = (F.fold.isin(set((0, 1, 2)) - {fold}) if fold < 3 else F.fold.isin((0, 1, 2))).to_numpy()
        te = (F.fold == fold).to_numpy()
        scale = StandardScaler().fit(X[tr]); xtr = scale.transform(X[tr]); xte = scale.transform(X[te])
        pca = PCA(n_components=64, svd_solver="randomized", random_state=20261025).fit(xtr)
        Utr = pca.transform(xtr); Ute = pca.transform(xte)
        score_cols = ["score", "s_kp", "s_kt"]
        mu = F.loc[tr, score_cols].mean().to_numpy(); sd = F.loc[tr, score_cols].std().to_numpy() + 1e-6
        Str = (F.loc[tr, score_cols].to_numpy() - mu) / sd
        Ste = (F.loc[te, score_cols].to_numpy() - mu) / sd
        site_n = F.loc[tr].groupby("region").size().to_dict()
        wt = np.array([1 / np.sqrt(site_n[r]) for r in F.loc[tr, "region"]])
        wt *= np.where(y[tr] == 1, (len(wt) - y[tr].sum()) / max(y[tr].sum(), 1), 1)
        clf = LogisticRegression(C=.01, max_iter=1000).fit(np.column_stack([Utr, Str]), y[tr], sample_weight=wt)
        base = LogisticRegression(C=.01, max_iter=1000).fit(Str, y[tr], sample_weight=wt)
        z[te] = clf.predict_proba(np.column_stack([Ute, Ste]))[:, 1]
        zb[te] = base.predict_proba(Ste)[:, 1]
        print("fold", fold, "n", int(te.sum()), "positive", int(y[te].sum()),
              "raw AP", round(average_precision_score(y[te], F.loc[te, "score"]), 4),
              "score-only AP", round(average_precision_score(y[te], zb[te]), 4),
              "DINO AP", round(average_precision_score(y[te], z[te]), 4),
              "DINO AUC", round(roc_auc_score(y[te], z[te]), 4), flush=True)
    F["p_dino"] = z; F["p_scoreonly"] = zb
    F.to_parquet(Path(__file__).with_name("sp_dino_candidate_probe.parquet"), index=False)

if __name__ == "__main__":
    main()
