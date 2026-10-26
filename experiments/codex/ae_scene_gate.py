"""Fold-honest scene gating for conservative aerial-film deployment.

The detector is frozen.  A small scene-level model estimates whether its top
proposals on an unseen film sheet are likely to contain a true damaged kurgan.
Every hyperparameter, including the gate threshold and per-scene cap, is
selected with inner geographic folds.  Predictions remain isolated by source
image as required by the organiser.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path("/home/nathan/Expedition Competition")
CW = ROOT / "Codex Work"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(CW)]

from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

KP = "kurgany_povrezhdennye"
CAPS = (0, 3, 5, 8, 10, 15, 20)
GATE_Q = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)


def f1(c):
    tp, fp, fn = c
    return 2 * tp / max(2 * tp + fn + .5 * fp, 1e-12)


def scene_features(g: pd.DataFrame) -> dict:
    s = np.sort(g.score.to_numpy())[::-1]
    kp = np.sort(g.s_kp.to_numpy())[::-1]
    kt = np.sort(g.s_kt.to_numpy())[::-1]
    out = {
        "n": len(g), "max": s[0], "mean": s.mean(), "std": s.std(),
        "n30": (s >= .30).sum(), "n40": (s >= .40).sum(),
        "n50": (s >= .50).sum(), "n60": (s >= .60).sum(),
        "kp_frac": (g.cls == KP).mean(),
        "kp_margin_mean": (g.s_kp - g.s_kt).mean(),
        "kp_margin_max": (g.s_kp - g.s_kt).max(),
    }
    for k in (3, 5, 10, 20, 50):
        out[f"top{k}"] = s[:k].mean()
        out[f"kp{k}"] = kp[:k].mean()
        out[f"margin{k}"] = (kp[:k] - kt[:k]).mean()
    return out


def counts(pred: pd.DataFrame, gt: dict) -> np.ndarray:
    _, r = score(fc(pred), gt, classes=[KP])
    v = r[KP]
    return np.array([v["tp"], v["fp"], v["fn"]], int)


def make_model(kind: str, seed: int):
    if kind == "logit":
        return make_pipeline(SimpleImputer(), RobustScaler(),
                             LogisticRegression(C=.25, class_weight="balanced",
                                                max_iter=2000, random_state=seed))
    return make_pipeline(SimpleImputer(), ExtraTreesClassifier(
        n_estimators=300, max_depth=2, min_samples_leaf=3,
        class_weight="balanced", max_features=.7, random_state=seed, n_jobs=-1))


def fit_predict(kind, Xtr, ytr, Xte, seed):
    # Constant behavior is important when an inner split has one target class.
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr[0]) if len(ytr) else 0.0)
    m = make_model(kind, seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def main():
    d = pd.read_parquet(CW / "ae_sheet_selfadapt_features.parquet")
    labels = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    labels = labels[(labels["mod"] == "Ae") & labels.region.isin(d.region.unique())]
    layered = layer_gt(labels)
    fold_of = d.groupby("region").fold.first().astype(int).to_dict()
    folds = sorted(set(fold_of.values()))

    scenes = []
    byscene = {}
    scene_gt = {}
    for key, g in d.groupby(["region", "scene"]):
        region, scene = key
        g = g.sort_values("score", ascending=False).copy()
        byscene[key] = g
        gg = layered[(layered.region == region) & (layered.scene == scene)]
        scene_gt[key] = to_gt_geojson(gg)
        row = {"region": region, "scene": scene, "fold": fold_of[region], **scene_features(g)}
        # Target is proposal usefulness at a conservative cap, not mere GT presence.
        p5 = g.head(5).copy(); p5["cls"] = KP
        c5 = counts(p5, scene_gt[key])
        row.update(tp5=c5[0], fp5=c5[1], fn5=c5[2], useful=int(c5[0] > 0))
        scenes.append(row)
    s = pd.DataFrame(scenes).sort_values(["fold", "region", "scene"]).reset_index(drop=True)
    feature_cols = [x for x in s.columns if x not in
                    {"region", "scene", "fold", "tp5", "fp5", "fn5", "useful"}]
    print("scenes", len(s), "useful", int(s.useful.sum()), "features", len(feature_cols))

    gt_fold = {f: to_gt_geojson(layered[layered.region.map(fold_of) == f]) for f in folds}
    kinds = ("logit", "trees")
    outer_total = np.zeros(3, int); choices = []; diagnostics = []
    for outer in folds:
        train_folds = [f for f in folds if f != outer]
        # Inner predictions are genuinely out-of-fold within the outer training set.
        inner_score = {k: np.full(len(s), np.nan) for k in kinds}
        for inner in train_folds:
            tr = s.fold.isin([f for f in train_folds if f != inner])
            va = s.fold == inner
            for kind in kinds:
                inner_score[kind][va] = fit_predict(
                    kind, s.loc[tr, feature_cols], s.loc[tr, "useful"],
                    s.loc[va, feature_cols], seed=1000 + 10 * outer + inner)

        candidates = []
        for kind in kinds:
            vals = inner_score[kind][s.fold.isin(train_folds)]
            thresholds = sorted(set(np.quantile(vals, GATE_Q).tolist() + [-1.0, 1.1]))
            for cap in CAPS:
                for th in thresholds:
                    tot = np.zeros(3, int)
                    for f in train_folds:
                        keep = (s.fold == f) & (inner_score[kind] >= th)
                        pred = []
                        for row in s[keep].itertuples():
                            q = byscene[(row.region, row.scene)].head(cap).copy()
                            q["cls"] = KP; pred.append(q)
                        pp = pd.concat(pred, ignore_index=True) if pred else d.iloc[:0].copy()
                        tot += counts(pp, gt_fold[f])
                    candidates.append((f1(tot), kind, cap, th, tot))
        best = max(candidates, key=lambda z: (z[0], -z[2]))
        _, kind, cap, th, inner_counts = best

        tr = s.fold.isin(train_folds); te = s.fold == outer
        p = fit_predict(kind, s.loc[tr, feature_cols], s.loc[tr, "useful"],
                        s.loc[te, feature_cols], seed=2000 + outer)
        pred = []
        selected = []
        for row, prob in zip(s[te].itertuples(), p):
            if prob >= th and cap:
                q = byscene[(row.region, row.scene)].head(cap).copy(); q["cls"] = KP
                pred.append(q); selected.append((row.region, row.scene, float(prob)))
        pp = pd.concat(pred, ignore_index=True) if pred else d.iloc[:0].copy()
        c = counts(pp, gt_fold[outer]); outer_total += c
        choices.append((outer, kind, cap, round(float(th), 4), tuple(inner_counts), tuple(c), len(selected)))
        diagnostics.extend((outer, *x) for x in selected)

    print("nested", f1(outer_total), tuple(outer_total))
    for x in choices: print("choice", x)
    print("selected scenes")
    for x in diagnostics: print(x)
    s.to_csv(CW / "ae_scene_gate_scene_table.csv", index=False)


if __name__ == "__main__":
    main()
