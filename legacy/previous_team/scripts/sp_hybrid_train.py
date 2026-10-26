"""Satellite hybrid, stage 2: LightGBM over physics candidates, judged under the official metric with honest thresholds.

    python scripts/sp_hybrid_train.py [--cands work/sp_hybrid/cands.parquet] [--no_cnn] [--tag v1]

* Model: LightGBM multiclass (0 = background, 1..5 = intact kurgan, damaged kurgan, hillfort, fortification,
  architecture) trained on the other folds (same 4 folds, 20 km clusters, as the U-Net run sp_r18l_v1).
* Output objects: for each class c, every candidate with p_c >= t_c becomes a disc of radius r (the blob scale) with
  confidence p_c. A candidate may be emitted under several classes (allowed by the metric; a miss costs 2x a false alarm).
* Evaluation: official matching per scene (scenes never overlap), pooled per class; thresholds chosen on the other folds.
  Reports honest and oracle F1 per class and the weighted final over satellite classes present.
Writes work/sp_hybrid/oof_<tag>.parquet, eval_<tag>.csv, deploy_<tag>.json and models_<tag>/ (fold + full models)."""
import argparse, json, os, sys
import numpy as np, pandas as pd
from shapely import wkb
from shapely.geometry import Point
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3.official.fastmatch import soft_match, prf
from koz3.official.compute_metrics import CLASS_WEIGHTS, normalize_class_name

ap = argparse.ArgumentParser(); ap.add_argument("--cands", default="work/sp_hybrid/cands.parquet"); ap.add_argument("--tag", default="v1")
ap.add_argument("--no_cnn", action="store_true"); ap.add_argument("--disc", type=float, default=1.0)
a = ap.parse_args()
OUT = os.path.dirname(a.cands)
CLASSES = ["курганы_целые", "курганы_поврежденные", "городища", "фортификации", "архитектуры"]
LAT = [normalize_class_name(c) for c in CLASSES]
C = pd.read_parquet(a.cands); C = C[C.fold >= 0].reset_index(drop=True)
LB = pd.read_parquet("work/sp/labels.parquet"); LB = LB[LB.scene.isin(set(C.scene))]
drop = {"x", "y", "tgt", "scene", "site", "fold", "r_m"}
feats = [c for c in C.columns if c not in drop and not (a.no_cnn and c.startswith("cnn"))]
print(f"{len(C)} candidates, {C.scene.nunique()} scenes, {len(feats)} features; positives per class:",
      C.tgt.value_counts().sort_index().to_dict(), flush=True)

import lightgbm as lgb
P = np.zeros((len(C), 6), np.float32)
params = dict(objective="multiclass", num_class=6, n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0, class_weight="balanced", n_jobs=4, verbose=-1)
os.makedirs(f"{OUT}/models_{a.tag}", exist_ok=True)
for f in sorted(C.fold.unique()):
    tr, te = (C.fold != f).values, (C.fold == f).values
    m = lgb.LGBMClassifier(**params).fit(C.loc[tr, feats], C.loc[tr, "tgt"])
    pr = m.predict_proba(C.loc[te, feats]); full = np.zeros((te.sum(), 6), np.float32); full[:, m.classes_] = pr; P[te] = full
    m.booster_.save_model(f"{OUT}/models_{a.tag}/fold{f}.txt")
    print(f"fold {f} done", flush=True)
for k in range(6):
    C[f"p{k}"] = P[:, k]
C.drop(columns=[c for c in C.columns if c.startswith("cnn")]).to_parquet(f"{OUT}/oof_{a.tag}.parquet")

THR = [1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]  # log grid: rare classes get tiny probabilities
G = {(s, c): [wkb.loads(g) for g in grp.wkb] for (s, c), grp in LB.groupby(["scene", "cls"])}
rows, deploy = [], {}
folds = sorted(C.fold.unique())
for ci, cls in enumerate(CLASSES):
    if not (LB.cls == cls).any():
        continue
    grid = {t: {f: np.zeros(3) for f in folds} for t in THR}
    for s, g in C.groupby("scene"):
        f = g.fold.iloc[0]; gts = G.get((s, cls), [])
        for t in THR:
            sel = g[g[f"p{ci + 1}"] >= t]
            preds = [(Point(x, y).buffer(r * a.disc, 16), p) for x, y, r, p in zip(sel.x, sel.y, sel.r_m, sel[f"p{ci + 1}"])]
            grid[t][f] += np.array(soft_match(preds, gts)[:3])
    tot = np.zeros(3); ff = []
    for f in folds:
        best = max(THR, key=lambda t: prf(*sum(grid[t][o] for o in folds if o != f))[2])
        tot += grid[best][f]; ff.append(round(prf(*grid[best][f])[2], 3))
    orc = max(THR, key=lambda t: prf(*sum(grid[t].values()))[2])
    rows.append(dict(cls=LAT[ci], F1_honest=round(prf(*tot)[2], 4), tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), per_fold=ff,
                     F1_oracle=round(prf(*sum(grid[orc].values()))[2], 4), thr_deploy=orc))
    deploy[LAT[ci]] = orc
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/eval_{a.tag}.csv", index=False)
fin = (R.F1_honest * R.cls.map(CLASS_WEIGHTS)).sum() / R.cls.map(CLASS_WEIGHTS).sum()
pd.set_option("display.width", 220); print(R.to_string(index=False)); print(f"final (satellite classes present, honest): {fin:.4f}")
json.dump(dict(thresholds=deploy, disc=a.disc, features=feats), open(f"{OUT}/deploy_{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# full model on all folds for deployment
lgb.LGBMClassifier(**params).fit(C[feats], C.tgt).booster_.save_model(f"{OUT}/models_{a.tag}/full.txt")
print("saved full model")
