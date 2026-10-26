"""Deployable pixel-LightGBM for object_poly (previous team's v3 recipe: 3 seeds, all regions, original renderings),
exported as booster JSON for koz3.gbm_np. Used 50/50 with our CNN v1 (nested OOF 0.353 vs 0.327 / 0.321 alone)."""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "kaggle_import/code/work/geo/objects_v3/units")
OUT = os.path.join(ROOT, "work/geo/objects_deploy")


def main():
    import lightgbm as lgb, pyarrow.parquet as pq
    from koz3.gbm_np import NumpyGBM
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{SRC}/*__orig.parquet"))
    meta = ("tgt", "uid", "region", "method", "render")
    feats = [c for c in pq.read_schema(files[0]).names if c not in meta]
    T = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    X, y = T[feats].to_numpy(np.float32), T.tgt.to_numpy()
    print(len(T), "pixels,", T.region.nunique(), "regions,", len(feats), "features")
    for sd, cs in ((0, 0.7), (1, 0.5), (2, 0.9)):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.7,
                               subsample_freq=1, colsample_bytree=cs, reg_lambda=2.0, random_state=sd, n_jobs=8, verbose=-1).fit(X, y)
        dump = m.booster_.dump_model(); json.dump(dump, open(f"{OUT}/obj_gbm_s{sd}.json", "w"))
        idx = np.random.default_rng(0).choice(len(X), 20000, replace=False)
        d = np.abs(NumpyGBM(dump).predict_proba(X[idx])[:, 1] - m.predict_proba(X[idx])[:, 1]).max()
        print(f"seed {sd}: numpy vs LightGBM max |Δp| {d:.2e}"); assert d < 1e-5
    json.dump(dict(features=feats, models=[f"obj_gbm_s{s}.json" for s in range(3)], w_cnn=0.5, thr=0.3, min_area_m2=0.5),
              open(f"{OUT}/objects.json", "w"), indent=1)


if __name__ == "__main__":
    main()
