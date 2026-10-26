"""Nested geographic OOF audit of a kurgan detector-score meta-model.

The base detectors' predictions are already OOF. Each outer fold gets a
LightGBM model trained only on the other regions. A two-fold/one-fold inner
cross-fit selects the probability cutoff without looking at the outer fold.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import _add_all, weighted
from koz3.score import score
from sp_assemble_oof import region_folds
from sp_fusion_oof import fc
from sp_kurgan_patch_features import names as patch_names

K = ("kurgany_povrezhdennye", "kurgany_tselye")
FEATURES = ["score", "m0", "m1", "m2", "support", "max_m", "std_m", "ratio_m0", "ratio_m1", "ratio_m2",
            "neighbor_30", "neighbor_100", "best_neighbor_30", "intact"]
USE_PATCH = "--patch" in sys.argv
if USE_PATCH:
    FEATURES += patch_names() + ["scene_found"]


def prepare():
    F = pd.read_parquet(Path(__file__).with_name("sp_agreement_oof.parquet"))
    F = F[F.cls.isin(K) & (F.score >= .08)].copy().reset_index(drop=True)
    if USE_PATCH:
        patch = pd.read_parquet(Path(__file__).with_name("sp_kurgan_patch_features.parquet"))
        assert len(patch) == len(F)
        assert (patch.region.to_numpy() == F.region.to_numpy()).all()
        assert (patch.cls.to_numpy() == F.cls.to_numpy()).all()
        assert np.allclose(patch.score.to_numpy(), F.score.to_numpy())
        F = pd.concat([F, patch[patch_names() + ["scene_found"]].reset_index(drop=True)], axis=1)
        F["scene_found"] = F.scene_found.astype(int)
    F["max_m"] = F[["m0", "m1", "m2"]].max(axis=1)
    F["std_m"] = F[["m0", "m1", "m2"]].std(axis=1)
    for c in ("m0", "m1", "m2"):
        F["ratio_" + c] = F[c] / F.max_m.clip(lower=1e-5)
    F["intact"] = (F.cls == K[1]).astype(int)
    F["neighbor_30"] = 0
    F["neighbor_100"] = 0
    F["best_neighbor_30"] = 0.0
    for reg, ids in F.groupby("region").groups.items():
        g = F.loc[ids]
        xy = g[["x", "y"]].to_numpy()
        tree = cKDTree(xy)
        scale = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
        near30 = tree.query_ball_point(xy, 30 * scale)
        near100 = tree.query_ball_point(xy, 100 * scale)
        F.loc[ids, "neighbor_30"] = [len(v) - 1 for v in near30]
        F.loc[ids, "neighbor_100"] = [len(v) - 1 for v in near100]
        scores = g.score.to_numpy()
        F.loc[ids, "best_neighbor_30"] = [max((scores[j] for j in v if j != i), default=0.0) for i, v in enumerate(near30)]
    RF = region_folds()
    F["fold"] = F.region.map(RF)
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.class_name.isin(K)]
    by_reg = {r: g.reset_index(drop=True) for r, g in L.groupby("region")}
    y = np.zeros(len(F), np.int8)
    for reg, ids in F.groupby("region").groups.items():
        gt = by_reg.get(reg)
        if gt is None or len(gt) == 0:
            continue
        tree = gt.sindex
        for i in ids:
            row = F.iloc[i]
            pt = Point(row.x, row.y)
            # Training label approximates the official representative-point
            # condition. Final evaluation below always uses the exact scorer.
            for j in tree.query(pt.buffer(5)):
                if gt.iloc[j].class_name != row.cls:
                    continue
                geom = gt.geometry.iloc[j]
                if (geom.geom_type == "Point" and geom.distance(pt) <= 2) or (geom.geom_type != "Point" and geom.covers(pt)):
                    y[i] = 1
                    break
    F["target"] = y
    return F


def fit_predict(train, test):
    model = lgb.LGBMClassifier(n_estimators=160, learning_rate=.035, num_leaves=9, max_depth=4,
                               min_child_samples=35, reg_lambda=5.0, subsample=.8, subsample_freq=1,
                               colsample_bytree=.9, verbosity=-1, n_jobs=1, random_state=20261023)
    model.fit(train[FEATURES], train.target)
    return model.predict_proba(test[FEATURES])[:, 1]


def surrogate(df, t, gt_counts):
    num = den = 0.0
    for cls, weight in ((K[0], 30), (K[1], 20)):
        g = df[df.cls == cls]
        pred = g.p_meta >= t
        tp = int((pred & (g.target == 1)).sum())
        fp = int((pred & (g.target == 0)).sum())
        total = gt_counts[cls]
        precision = tp / (tp + .5 * fp + 1e-8)
        recall = tp / (total + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        num += weight * f1; den += weight
    return num / den


def main():
    F = prepare()
    folds = sorted(F.fold.unique())
    thresholds = np.array([.02, .03, .04, .05, .07, .10, .13, .16, .20, .25, .30, .40])
    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    labels = labels[(labels["mod"] == "SpOr") & labels.region.isin(set(F.region))]
    labels["fold"] = labels.region.map(region_folds())
    chosen = {}
    F["p_meta"] = np.nan
    for outer in folds:
        inner = [f for f in folds if f != outer]
        chunks = []
        for validation in inner:
            tr = F[F.fold.isin([f for f in inner if f != validation])]
            va = F[F.fold == validation].copy()
            va["p_meta"] = fit_predict(tr, va)
            chunks.append(va)
        inner_oof = pd.concat(chunks)
        label_inner = labels[labels.fold.isin(inner)]
        gt_counts = {cls: int((label_inner.class_name == cls).sum()) for cls in K}
        chosen[int(outer)] = float(max(thresholds, key=lambda t: surrogate(inner_oof, t, gt_counts)))
        tr = F[F.fold != outer]
        te = F[F.fold == outer]
        F.loc[te.index, "p_meta"] = fit_predict(tr, te)
    assert F.p_meta.notna().all()
    F.to_parquet(Path(__file__).with_name("sp_kurgan_meta_patch_predictions.parquet" if USE_PATCH else "sp_kurgan_meta_predictions.parquet"), index=False)
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    regions = set(F.region)
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    GT = {f: to_gt_geojson(L[L.region.map(region_folds()) == f]) for f in folds}
    all_predictions = pd.read_parquet(Path(__file__).with_name("sp_agreement_oof.parquet"))
    C = {"baseline": {}, "meta": {}}
    for fold in folds:
        regions_f = set(F[F.fold == fold].region)
        b = all_predictions[(all_predictions.region.isin(regions_f)) & (all_predictions.score >= .25)]
        selected = F[(F.fold == fold) & (F.p_meta >= chosen[int(fold)])]
        nonk = b[~b.cls.isin(K)]
        for name, pred in (("baseline", b), ("meta", pd.concat([nonk, selected], ignore_index=True))):
            _, result = score(fc(pred), GT[fold])
            C[name][int(fold)] = {k: (v["tp"], v["fp"], v["fn"]) for k, v in result.items()}
    summary = {"regions": len(regions), "candidate_count": len(F), "positive_count": int(F.target.sum()),
               "folds": {str(f): {"threshold": chosen[int(f)], "baseline": weighted(C["baseline"][int(f)]),
                                   "meta": weighted(C["meta"][int(f)])} for f in folds},
               "baseline": weighted(_add_all(C["baseline"].values())),
               "meta": weighted(_add_all(C["meta"].values())),
               "class_counts": {arm: {k: list(map(int, v)) for k, v in _add_all(C[arm].values()).items()} for arm in C}}
    (Path(__file__).with_name("sp_kurgan_meta_patch_oof.json") if USE_PATCH else Path(__file__).with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
