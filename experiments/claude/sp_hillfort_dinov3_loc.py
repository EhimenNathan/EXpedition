"""Stage 2 of the satellite hillfort localiser: where inside a satellite scene should the hillfort point go?

Per token (32 px patch of the whole-scene view): P(token centre inside a hillfort) from
  prior   logistic on position only (u, v, u², v², r, r²; hillforts sit near the scene centre)
  probe   logistic on DINOv3-SAT tokens (PCA 128 fitted on the training folds)
  joint   logistic on tokens + position (Bayesian combination of appearance likelihood and framing prior)
trained on the other folds only (koz3.sp_prep.site_folds: 20 km site clusters, 4 folds). Per scene the map is smoothed
(Gaussian σ = 1 token) and its argmax is the emitted point; HIT = point inside a true hillfort polygon. Reported on scenes
that contain a hillfort, against the scene centre (the deployed fallback). Scene-level AUC (max of the map) as a
by-product.
  python experiments/claude/sp_hillfort_dinov3_loc.py [tokens=work/hf_dinov3/tokens.npz]
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)


def pos_feats(pos):
    u, v = pos[:, 0], pos[:, 1]; r = np.hypot(u, v)
    return np.c_[u, v, u * u, v * v, r, r * r]


def decode(p, gh, gw, sigma=1.0):
    from scipy.ndimage import gaussian_filter
    m = gaussian_filter(p.reshape(gh, gw), sigma, mode="nearest")
    return int(np.argmax(m)), float(m.max())


def main(tokens="work/hf_dinov3/tokens.npz", pca_dim=128, C=0.05):
    import geopandas as gpd
    from shapely.geometry import Point
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from koz3.sp_prep import site_folds
    Z = np.load(os.path.join(ROOT, tokens), allow_pickle=True)
    tok, y, pos, sid, grid, meta = Z["tok"].astype(np.float32), Z["y"], Z["pos"], Z["sid"], Z["grid"], Z["meta"]
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]
    fold_of = site_folds(L)
    regions = np.array([m[0] for m in meta]); sfold = np.array([fold_of[r] for r in regions]); tfold = sfold[sid]
    HF = {r: g.geometry.union_all() for r, g in L[L.class_name == "gorodishcha"].groupby("region")}
    print(f"{len(meta)} scenes, {len(tok)} tokens, {int(y.sum())} hillfort tokens, folds {np.bincount(sfold)}", flush=True)
    P = {k: np.zeros(len(tok)) for k in ("prior", "probe", "joint")}
    for f in np.unique(sfold):
        tr, te = tfold != f, tfold == f
        pca = PCA(pca_dim, random_state=0).fit(tok[tr][::4])
        A = pca.transform(tok); A /= A[tr].std(0) + 1e-6
        Q = pos_feats(pos)
        for k, X in (("prior", Q), ("probe", A), ("joint", np.c_[A, Q])):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=C if k != "prior" else 1.0, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            P[k][te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        print(f"  fold {f} done", flush=True)
    for k in P:
        print(f"token AUC {k:6s} {roc_auc_score(y, P[k]):.3f}")
    rows = []
    for s, (gh, gw) in enumerate(grid):
        idx = np.flatnonzero(sid == s); r = regions[s]
        has = bool(y[idx].any())
        cx, cy = pos[idx][:, 2].mean(), pos[idx][:, 3].mean()          # scene centre in EPSG:3857 (token-centre mean)
        row = dict(scene=s, region=r, fold=sfold[s], has=has, centre=bool(r in HF and HF[r].contains(Point(cx, cy))))
        for k in P:
            a, mx = decode(P[k][idx], gh, gw)
            row[k] = bool(y[idx][a]); row[k + "_max"] = mx
        rows.append(row)
    import pandas as pd
    D = pd.DataFrame(rows); H = D[D.has]
    print(f"\nscenes containing a hillfort: {len(H)} (regions {H.region.nunique()})")
    print("HIT rate (emitted point inside the hillfort):  " + "  ".join(f"{k} {H[k].mean():.3f}" for k in ("centre", "prior", "probe", "joint")))
    for f in sorted(H.fold.unique()):
        h = H[H.fold == f]
        print(f"  fold {f} ({len(h)} scenes): " + "  ".join(f"{k} {h[k].mean():.2f}" for k in ("centre", "prior", "probe", "joint")))
    print("per region (first scene): joint vs centre  wins", int(((H.joint) & (~H.centre)).sum()), " losses", int(((~H.joint) & (H.centre)).sum()))
    for k in ("probe", "joint"):
        print(f"scene AUC hillfort present ({k} map max): {roc_auc_score(D.has, D[k + '_max']):.3f}")
    D.to_csv(os.path.join(ROOT, "work/hf_dinov3/loc_oof.csv"), index=False)
    np.savez(os.path.join(ROOT, "work/hf_dinov3/token_oof.npz"), **P)


if __name__ == "__main__":
    main(*sys.argv[1:])
