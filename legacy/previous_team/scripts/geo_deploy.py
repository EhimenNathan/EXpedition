"""Train the deployable geophysics models on ALL training regions and write them into the inference package.

    python scripts/geo_deploy.py [--exclude 200_]

finds_points: one LightGBM (the honest-best 'base' recipe of scripts/geo_finds_v3.py) on every region's original
  rendering; probability threshold = best over all out-of-fold regions (work/geo/finds_v3/oof.parquet, p_base).
object_poly: three LightGBM seeds (scripts/geo_objects_v3.py recipe) on every region's original-rendering pixels;
  per-method post-processing (sigma, threshold, min area) = best over all out-of-fold regions (counts_orig.csv).
Deploy choices use all out-of-fold results (the honest estimate of that choice is the leave-one-region-out number
reported in docs/WORKFLOW.md section 6). Outputs: inference/models/geo/{finds_lgb.txt, finds.json, obj_s*.txt, objects.json}"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3.official.fastmatch import soft_match, prf
from shapely.geometry import Point

ap = argparse.ArgumentParser(); ap.add_argument("--exclude", default="200_"); a = ap.parse_args()
OUT = "inference/models/geo"; os.makedirs(OUT, exist_ok=True)
excl = tuple(x for x in a.exclude.split(",") if x)

# ---------------- finds
F = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("work/geo/finds_v3/units/*__orig.parquet"))], ignore_index=True)
if excl:
    F = F[~F.region.str.startswith(excl)]
feats = [c for c in F.columns if c not in ("x", "y", "tgt", "uid", "region", "render")]
m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.7, reg_lambda=2.0, random_state=0, n_jobs=4, verbose=-1).fit(F[feats], F.tgt)
m.booster_.save_model(f"{OUT}/finds_lgb.txt")
O = pd.read_parquet("work/geo/finds_v3/oof.parquet"); GT = {k: np.array(v) for k, v in json.load(open("work/geo/finds_v3/gt.json", encoding="utf-8")).items()}
grid = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]; tot = {t: np.zeros(3) for t in grid}
for uid, g in O.groupby("uid"):
    gts = [Point(x, y) for x, y in GT[uid]]
    for t in grid:
        s = g[g.p_base >= t]; tot[t] += soft_match([(Point(x, y).buffer(0.25, 8), p) for x, y, p in zip(s.x, s.y, s.p_base)], gts)[:3]
thr = max(grid, key=lambda t: prf(*tot[t])[2])
json.dump(dict(features=feats, threshold=thr, pred_radius_m=0.25, gsd=0.1, oof_f1_at_threshold=round(prf(*tot[thr])[2], 4),
               trained_rows=len(F), regions=sorted(F.region.unique())), open(f"{OUT}/finds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"finds: {len(F)} rows, {F.region.nunique()} regions, threshold {thr} (OOF F1 at it {prf(*tot[thr])[2]:.4f})", flush=True)
del F

# ---------------- objects
files = [p for p in sorted(glob.glob("work/geo/objects_v3/units/*__orig.parquet")) if not os.path.basename(p).startswith(excl or ("\0",))]
import pyarrow.parquet as pq
meta = ("tgt", "uid", "region", "method", "render")
ofeats = [c for c in pq.read_schema(files[0]).names if c not in meta]
n = sum(pq.read_metadata(p).num_rows for p in files)
X = np.empty((n, len(ofeats)), np.float32); y = np.empty(n, np.int8); i0 = 0
for p in files:
    t = pd.read_parquet(p); k = len(t); X[i0:i0 + k] = t[ofeats].to_numpy(np.float32); y[i0:i0 + k] = t.tgt.to_numpy(); i0 += k
for sd, cs in ((0, 0.7), (1, 0.5), (2, 0.9)):
    lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.7, subsample_freq=1,
                       colsample_bytree=cs, reg_lambda=2.0, random_state=sd, n_jobs=4, verbose=-1).fit(X, y).booster_.save_model(f"{OUT}/obj_s{sd}.txt")
R = pd.read_csv("work/geo/objects_v3/counts_orig.csv"); key = ["sigma", "thr", "min_area"]; post = {}
for meth, g in R.groupby("method"):
    s = g.groupby(key)[["tp", "fp", "fn"]].sum(); b = max(s.index, key=lambda k: prf(*s.loc[k].values)[2])
    post[meth] = dict(sigma=float(b[0]), thr=float(b[1]), min_area=float(b[2]), oof_f1=round(prf(*s.loc[b].values)[2], 4))
json.dump(dict(features=ofeats, models=[f"obj_s{sd}.txt" for sd in range(3)], post=post, trained_rows=int(n)),
          open(f"{OUT}/objects.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"objects: {n} pixels, post-processing {post}")
