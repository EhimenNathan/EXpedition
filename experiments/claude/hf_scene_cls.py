"""Out-of-fold scene-level hillfort presence classifier from whole-scene DINOv3 tokens (mean + std pooled, PCA 64 fitted
on the training folds, L2 logistic C 0.05), folds koz3.sp_prep.site_folds of the data type; writes p_scene into the
localiser's OOF point table.
  KOZ3_MOD=Li python experiments/claude/hf_scene_cls.py <points.csv> <tokens.npz> [<tokens2.npz> ...]
"""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)


def main(csv, *token_files):
    import geopandas as gpd, pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from koz3.sp_prep import site_folds
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == os.environ.get("KOZ3_MOD", "SpOr")]
    fo = site_folds(L); feats = []
    for fn in token_files:
        Z = np.load(os.path.join(ROOT, fn), allow_pickle=True); tok = Z["tok"].astype(np.float32); sid = Z["sid"]
        feats.append(np.stack([np.r_[tok[sid == s].mean(0), tok[sid == s].std(0)] for s in range(len(Z["meta"]))]))
    X = np.concatenate(feats, 1); meta = Z["meta"]
    has = np.array([Z["y"][Z["sid"] == s].any() for s in range(len(meta))]); f = np.array([fo[m[0]] for m in meta])
    p = np.zeros(len(X))
    for k in np.unique(f):
        tr, te = f != k, f == k
        pca = PCA(min(64, tr.sum() - 1), random_state=0).fit(X[tr]); sc = StandardScaler().fit(pca.transform(X[tr]))
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(pca.transform(X[tr])), has[tr])
        p[te] = m.predict_proba(sc.transform(pca.transform(X[te])))[:, 1]
    D = pd.read_csv(os.path.join(ROOT, csv)); assert len(D) == len(p) and (D.tif.values == np.array([m[1] for m in meta])).all()
    D["p_scene"] = p; D.to_csv(os.path.join(ROOT, csv), index=False)
    print(f"scene AUC classifier {roc_auc_score(has, p):.3f} | map max {roc_auc_score(has, D.map_max):.3f} | mean {roc_auc_score(has, (p + D.map_max) / 2):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
