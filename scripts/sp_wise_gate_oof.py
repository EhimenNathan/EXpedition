"""Nested geographic gate for the promising but unstable WiSE-FT satellite member.

WiSE-FT alpha=0.2 improves three satellite folds but regresses fold 0.  This
test asks whether that behavior is predictable from label-free region signals:
the shift in candidate counts/scores/positions, scene-presence score and scene
geometry.  A tiny ridge, stump, or strongly regularised MLP predicts the
per-region utility of adding WiSE as a seventh fusion member.

For every outer fold, model family and gate threshold are selected using
inner leave-one-geographic-fold-out predictions on the other three folds.
The official metric is used for selection and final evaluation.  No held-out
labels enter feature extraction, fitting, model selection, or thresholding.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]


def estimators():
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def scaled(model):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

    return {
        "ridge1": scaled(Ridge(alpha=1.0)),
        "ridge10": scaled(Ridge(alpha=10.0)),
        "ridge100": scaled(Ridge(alpha=100.0)),
        "huber": scaled(HuberRegressor(epsilon=1.5, alpha=1.0, max_iter=1000)),
        "stump": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, max_depth=1, min_samples_leaf=8,
                                  max_features=0.7, random_state=20261029, n_jobs=4),
        ),
        "rf2": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, max_depth=2, min_samples_leaf=8,
                                  max_features=0.7, random_state=20261029, n_jobs=4),
        ),
        "mlp8": scaled(MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", alpha=3.0,
                                     learning_rate_init=0.003, max_iter=2000,
                                     early_stopping=False, random_state=20261029)),
    }


def region_features(base_member, wise_member, scene, geometry, regions):
    """One row per region; every column is available without labels."""
    rows = []
    classes = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii")
    for region in regions:
        b = base_member[base_member.region == region]
        w = wise_member[wise_member.region == region]
        out = {"region": region}
        for tag, frame in (("ft", b), ("wise", w)):
            score = frame.score.to_numpy(float)
            out[f"{tag}_n"] = len(frame)
            for t in (0.1, 0.2, 0.3, 0.5):
                out[f"{tag}_n{int(t*10)}"] = int((score >= t).sum())
            for q in (0.5, 0.75, 0.9, 0.99):
                out[f"{tag}_q{int(q*100)}"] = float(np.quantile(score, q)) if len(score) else 0.0
            for cls in classes:
                z = frame[frame.cls == cls].score.to_numpy(float)
                key = cls.split("_")[0]
                out[f"{tag}_{key}_n"] = len(z)
                out[f"{tag}_{key}_q90"] = float(np.quantile(z, .9)) if len(z) else 0.0

        # Candidate-set disagreement is a direct, label-free measure of how
        # strongly interpolation changed the model on this unseen region.
        if len(b) and len(w):
            bt = cKDTree(b[["x", "y"]].to_numpy(float))
            dist, idx = bt.query(w[["x", "y"]].to_numpy(float), k=1)
            near = dist <= 8.0
            out["match8"] = float(near.mean())
            out["match_dist_med"] = float(np.median(dist))
            out["match_score_abs"] = float(np.mean(np.abs(w.score.to_numpy(float)[near] - b.score.to_numpy(float)[idx[near]]))) if near.any() else 1.0
        else:
            out.update(match8=0.0, match_dist_med=999.0, match_score_abs=1.0)

        s = scene[scene.region == region]
        out["scene_p_mean"] = float(s.p.mean()) if len(s) else 0.0
        out["scene_p_max"] = float(s.p.max()) if len(s) else 0.0
        g = geometry.get(region, [])
        out["scene_count"] = len(g)
        out["scene_small_share"] = float(np.mean([sm for _, _, _, sm in g])) if g else 0.0
        out["scene_halfwidth_max"] = float(max([h for _, _, h, _ in g], default=0.0))
        rows.append(out)
    f = pd.DataFrame(rows).set_index("region")
    # Explicit deltas make the low-capacity models' task well conditioned.
    for c in [c for c in f if c.startswith("wise_")]:
        b = "ft_" + c[5:]
        if b in f:
            f["delta_" + c[5:]] = f[c] - f[b]
    return f.replace([np.inf, -np.inf], np.nan)


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import _add_all, weighted
    from sp_assemble_oof import models, region_folds
    from sp_dual_oof import K, fuse_cls
    from sp_fusion4_oof import gate
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from robust_accept import report, site_counts

    names = ["dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3"]
    M = models(extra=names[3:] + ["dfnodn_wise02"])
    regions = sorted(set.intersection(*[set(M[n].region) for n in names + ["dfnodn_wise02"]]))
    rf = region_folds(); folds = sorted({rf[r] for r in regions})
    labels = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    labels = labels[(labels["mod"] == "SpOr") & labels.region.isin(regions)]
    gt = {r: to_gt_geojson(labels[labels.region == r]) for r in regions}
    scene = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet"))
    geom = scene_geometry()

    fused = {
        "base": fuse_cls([M[n][M[n].region.isin(regions)] for n in names]),
        "wise": fuse_cls([M[n][M[n].region.isin(regions)] for n in names + ["dfnodn_wise02"]]),
    }
    print("parity inputs:", len(labels), "GT labels; fused rows", {k: len(v) for k, v in fused.items()}, flush=True)

    def pipeline(which, region):
        f = fused[which]
        f = f[f.region == region].copy()
        t = 0.25
        f = f[(f.score >= np.where(f.cls == "arkhitektury", 0.6 * t, t))]
        keep = []
        for row in f.itertuples():
            d = min([(np.hypot(row.x-cx, row.y-cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(region, [])] or [(0.0, True)])
            keep.append(d[1] or d[0] <= 0.5 or row.score >= 1.4 * t)
        f = f[np.asarray(keep, bool)]
        ex = [(row.region, K[1] if row.cls == K[0] else K[0], row.score*.99, row.x, row.y)
              for row in f[f.cls.isin(K)].itertuples()
              if (row.s_kt if row.cls == K[0] else row.s_kp) >= .4]
        f = pd.concat([f[["region", "cls", "score", "x", "y"]],
                       pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        return fc(gate(f, scene[scene.region == region]))

    probe = regions[0]
    print("parity probe", probe, "GT", len(gt[probe]["features"]),
          "base predictions", len(pipeline("base", probe)["features"]), flush=True)
    counts = site_counts(lambda s, r: pipeline(s, r), ("base", "wise"), regions, gt)
    target = pd.Series({r: weighted(counts["wise"][r]) - weighted(counts["base"][r]) for r in regions})
    x = region_features(M["dfnodn"], M["dfnodn_wise02"], scene, geom, regions)
    cols = list(x.columns)
    print(f"{len(regions)} regions, {len(cols)} label-free features; site target up/down/same "
          f"{int((target>0).sum())}/{int((target<0).sum())}/{int((target==0).sum())}", flush=True)

    families = estimators()
    chosen = {}
    prediction = pd.Series(np.nan, index=regions, dtype=float)
    for outer in folds:
        train = [r for r in regions if rf[r] != outer]
        test = [r for r in regions if rf[r] == outer]
        candidates = []
        for name in families:
            inner_pred = pd.Series(np.nan, index=train, dtype=float)
            for inner in [f for f in folds if f != outer]:
                tr = [r for r in train if rf[r] != inner]
                va = [r for r in train if rf[r] == inner]
                model = estimators()[name]
                model.fit(x.loc[tr, cols], target.loc[tr])
                inner_pred.loc[va] = model.predict(x.loc[va, cols])
            thresholds = sorted(set([-np.inf, 0.0, np.inf] + list(np.quantile(inner_pred, np.linspace(0, 1, 21)))))
            for threshold in thresholds:
                total = _add_all(counts["wise" if inner_pred[r] >= threshold else "base"][r] for r in train)
                candidates.append((weighted(total), name, float(threshold)))
        _, name, threshold = max(candidates, key=lambda z: z[0])
        model = estimators()[name]
        model.fit(x.loc[train, cols], target.loc[train])
        prediction.loc[test] = model.predict(x.loc[test, cols])
        chosen[outer] = (name, threshold)
        print(f"outer {outer}: chose {name} gate {threshold:+.5f}; WiSE on "
              f"{sum(prediction[r] >= threshold for r in test)}/{len(test)} regions", flush=True)

    mixed = {}
    fold_change = {}
    for fold in folds:
        name, threshold = chosen[fold]
        test = [r for r in regions if rf[r] == fold]
        mixed.update({r: counts["wise" if prediction[r] >= threshold else "base"][r] for r in test})
        fold_change[fold] = weighted(_add_all(mixed[r] for r in test)) - weighted(_add_all(counts["base"][r] for r in test))
    base_score = weighted(_add_all(counts["base"].values()))
    wise_score = weighted(_add_all(counts["wise"].values()))
    mixed_score = weighted(_add_all(mixed.values()))
    print(f"base {base_score:.5f} | unconditional WiSE {wise_score:.5f} | nested gated {mixed_score:.5f}", flush=True)
    print("fold changes", {f: round(v, 5) for f, v in fold_change.items()}, flush=True)
    report("nested label-free WiSE gate", counts["base"], mixed)


if __name__ == "__main__":
    main()
