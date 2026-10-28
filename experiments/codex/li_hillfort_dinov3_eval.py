"""Geographic OOF DINOv3-SAT localizer and presence probe on LiDAR relief."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.claude.sp_hillfort_dinov3_eval import DIMS, CS, oof_maps, per_scene


def main(src="work/hf_dinov3/li_tokens512.npz", out="work/hf_dinov3/li_loc_points_oof.csv"):
    import geopandas as gpd
    from scipy.ndimage import gaussian_filter
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z = dict(np.load(ROOT / src, allow_pickle=True))
    meta, sid = z["meta"], z["sid"]
    folds = np.asarray([int(m[5]) for m in meta])
    token_folds = folds[sid]
    probs = oof_maps(z, token_folds)
    maps = []
    for p in probs.values():
        maps.append([per_scene(z, p, s)[1] for s in range(len(meta))])
    ensemble = [np.mean([q[s] for q in maps], axis=0) for s in range(len(meta))]

    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    positive = set(labels[(labels["mod"] == "Li") & (labels.class_name == "gorodishcha")].region)
    has = np.asarray([str(m[0]) in positive for m in meta])
    scene_x = np.stack([np.r_[z["tok"][sid == s].astype(np.float32).mean(0),
                               z["tok"][sid == s].astype(np.float32).std(0)] for s in range(len(meta))])
    p_scene = np.zeros(len(meta))
    for fold in sorted(set(folds)):
        tr, te = folds != fold, folds == fold
        dim = min(16, int(tr.sum()) - 2)
        pca = PCA(dim, random_state=0).fit(scene_x[tr])
        a = pca.transform(scene_x)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=.05, class_weight="balanced", max_iter=3000))
        clf.fit(a[tr], has[tr].astype(int))
        p_scene[te] = clf.predict_proba(a[te])[:, 1]

    rows = []
    for s, m in enumerate(meta):
        idx = np.flatnonzero(sid == s)
        g = gaussian_filter(ensemble[s], .5, mode="nearest")
        a = int(np.argmax(g))
        rows.append((m[0], z["pos"][idx][a, 2], z["pos"][idx][a, 3], float(g.max()),
                     bool(has[s]), bool(z["y"][idx][a]), int(folds[s]), float(p_scene[s]),
                     int(z["y"][idx].sum())))
    d = pd.DataFrame(rows, columns=["region", "x3857", "y3857", "map_max", "has_hillfort", "hit",
                                      "fold", "p_scene", "positive_tokens"])
    path = ROOT / out; path.parent.mkdir(parents=True, exist_ok=True); d.to_csv(path, index=False)
    pos = d[d.has_hillfort]
    print("positive sites", len(pos), "covered by token centers", int((pos.positive_tokens > 0).sum()))
    print("center-grid localizer hit", round(float(pos.hit.mean()), 4), "hits", int(pos.hit.sum()))
    print("presence AUC", round(roc_auc_score(d.has_hillfort, d.p_scene), 4))
    print("presence fold", [round(roc_auc_score(d[d.fold == f].has_hillfort, d[d.fold == f].p_scene), 4)
                            for f in sorted(d.fold.unique())])
    print("saved", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
