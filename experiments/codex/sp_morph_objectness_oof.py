"""Prospective fold-3 test of optical-morphology kurgan candidate filtering.

Only folds 0,1,2 choose the fixed score threshold; fold 3 is untouched until
the final evaluation. Original detector OOF remains site-disjoint. LightGBM
uses optical features and raw scores, with site-balanced sample weights.
"""
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.nested import weighted, _add_all
from koz3.score import score
from sp_assemble_oof import models, region_folds
from sp_dual_oof import fuse_cls
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry

NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
K = ("kurgany_povrezhdennye", "kurgany_tselye")
THRESHOLDS = (.1, .15, .2, .25, .3, .35, .4, .5, .6, .7)


def fit_predict(train, test, columns):
    tr = train[train.fold.isin((0, 1, 2))]
    y = tr.y_obj.to_numpy()
    counts = tr.groupby("region").size().to_dict()
    site_w = np.array([1 / np.sqrt(counts[r]) for r in tr.region])
    class_w = np.where(y == 1, (len(y) - y.sum()) / max(1, y.sum()), 1.0)
    weight = site_w * class_w
    model = lgb.LGBMClassifier(n_estimators=220, learning_rate=.05, num_leaves=15,
                               max_depth=4, min_child_samples=60, reg_lambda=5.0,
                               colsample_bytree=.8, subsample=.8, subsample_freq=1,
                               verbosity=-1, n_jobs=2, random_state=20261025)
    model.fit(tr[columns], y, sample_weight=weight)
    return model.predict_proba(test[columns])[:, 1], model


def main():
    T = pd.read_parquet(Path(__file__).with_name("sp_morph_features.parquet"))
    print("optical features", len(T), "positives", int(T.y_obj.sum()),
          "folds", T.fold.value_counts().to_dict(), flush=True)
    columns = [c for c in T if c.startswith(("gray_", "rgb", "radial_", "angular_", "gradient_", "linear_", "tangential_"))]
    columns += ["score", "s_kp", "s_kt", "valid_frac_48", "center_gray"]
    T["p_obj"] = np.nan
    for f in (0, 1, 2):
        tr = T[T.fold.isin(set((0, 1, 2)) - {f})]
        te = T[T.fold == f]
        p, _ = fit_predict(tr, te, columns)
        T.loc[te.index, "p_obj"] = p
        print("inner fold", f, "p median", float(np.median(p)), flush=True)
    te = T[T.fold == 3]
    T.loc[te.index, "p_obj"], final_model = fit_predict(T[T.fold.isin((0, 1, 2))], te, columns)
    print("fold3 p median", float(T[T.fold == 3].p_obj.median()), flush=True)
    # A diagnostic only: optical feature contribution vs score-only ranking.
    print("top features", sorted(zip(columns, final_model.feature_importances_), key=lambda z: -z[1])[:15], flush=True)
    assert T.p_obj.notna().all()
    M = models(extra=list(NAMES[3:]))
    regions = sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
    RF = region_folds()
    F = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES])
    F = F[(F.cls.isin(K) & (F.score >= .08)) |
          ((F.cls == "arkhitektury") & (F.score >= .15)) |
          ((~F.cls.isin(K + ("arkhitektury",))) & (F.score >= .25))].copy()
    T = T[["region", "cls", "score", "x", "y", "p_obj"]]
    F = F.merge(T, how="left", on=["region", "cls", "score", "x", "y"], validate="one_to_one")
    geom = scene_geometry()
    keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])]
                or [(0.0, True)])
        keep.append(d[1] or d[0] <= .5 or r.score >= .35)
    F = F[np.array(keep)]
    S = pd.read_parquet(ROOT / "work/spor/oof_scene_cls.parquet")
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in range(4)}

    def make(fold, threshold=None):
        g = F[F.region.map(RF) == fold].copy()
        if threshold is None:
            mask = (~g.cls.isin(K)) | (g.score >= .25)
        else:
            mask = (~g.cls.isin(K)) | (g.p_obj >= threshold) | (g.p_obj.isna() & (g.score >= .25))
        g = g[mask].copy()
        if threshold is not None:
            q = g.cls.isin(K) & g.p_obj.notna()
            g.loc[q, "score"] = g.loc[q, "p_obj"]
        extra = []
        for r in g[g.cls.isin(K)].itertuples():
            other = K[1] if r.cls == K[0] else K[0]
            if (r.s_kt if other == K[1] else r.s_kp) >= .4:
                extra.append((r.region, other, r.score * .99, r.x, r.y))
        g = g[["region", "cls", "score", "x", "y"]]
        if extra:
            g = pd.concat([g, pd.DataFrame(extra, columns=g.columns)], ignore_index=True)
        return fc(gate(g, S[S.region.map(RF) == fold]))

    def scored_counts(fc_, target):
        return {c: (v["tp"], v["fp"], v["fn"]) for c, v in score(fc_, target)[1].items()}

    baseline = {f: scored_counts(make(f), gt[f]) for f in range(4)}
    print("baseline parity", weighted(_add_all(baseline.values())), flush=True)
    C = {t: {f: scored_counts(make(f, t), gt[f]) for f in range(4)} for t in THRESHOLDS}
    base_inner = weighted(_add_all(baseline[f] for f in (0, 1, 2)))
    chosen = max(THRESHOLDS, key=lambda t: weighted(_add_all(C[t][f] for f in (0, 1, 2))))
    print("base inner", base_inner, "chosen", chosen, flush=True)
    for t in THRESHOLDS:
        print("threshold", t, "inner", weighted(_add_all(C[t][f] for f in (0, 1, 2))),
              "inner folds", [weighted(C[t][f]) for f in (0, 1, 2)], flush=True)
    print("UNTOUCHED_FOLD3", "base", weighted(baseline[3]), "morph", weighted(C[chosen][3]),
          "delta", weighted(C[chosen][3]) - weighted(baseline[3]), flush=True)
    print("all pooled", weighted(_add_all(baseline.values())),
          weighted(_add_all(C[chosen].values())), flush=True)
    (Path(__file__).with_name("sp_morph_objectness_verdict.json")).write_text(json.dumps({
        "chosen_threshold_on_folds012": chosen,
        "base_inner": base_inner,
        "morph_inner": weighted(_add_all(C[chosen][f] for f in (0, 1, 2))),
        "base_fold3": weighted(baseline[3]),
        "morph_fold3": weighted(C[chosen][3]),
        "base_all": weighted(_add_all(baseline.values())),
        "morph_all": weighted(_add_all(C[chosen].values())),
    }, indent=2))

if __name__ == "__main__":
    main()
