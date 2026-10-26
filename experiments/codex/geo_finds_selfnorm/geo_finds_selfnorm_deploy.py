#!/usr/bin/env python3
"""Train/export the geographically validated raw+within-map-rank MagGr finds model."""
from pathlib import Path
import glob, json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"kaggle_import/code/work/geo/finds_v3"
OUT=ROOT/"work/geo/finds_selfnorm_deploy"; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from koz3.gbm_np import NumpyGBM

DROP={"x","y","tgt","uid","region","render"}

def add_rank(df):
    parts=[]
    for _,g in df.groupby("uid",sort=False):
        g=g.copy(); cols=[c for c in g if c not in DROP]
        q=g[cols].rank(method="average",pct=True).astype(np.float32)
        q.columns=["rank__"+c for c in cols]
        parts.append(pd.concat([g,q],axis=1))
    return pd.concat(parts,ignore_index=True)

def main():
    F=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(SRC/"units/*__orig.parquet")))],ignore_index=True)
    F=F.drop_duplicates(["uid","x","y"]).reset_index(drop=True)
    F=add_rank(F)
    raw=[c for c in F if c not in DROP and not c.startswith("rank__")]
    feats=raw+["rank__"+c for c in raw]
    m=lgb.LGBMClassifier(n_estimators=500,learning_rate=.025,num_leaves=25,min_child_samples=30,
        subsample=.8,subsample_freq=1,colsample_bytree=.65,reg_lambda=3.,random_state=0,n_jobs=8,verbose=-1)
    m.fit(F[feats],F.tgt); dump=m.booster_.dump_model()
    json.dump(dump,open(OUT/"finds_gbm.json","w"))
    sample=F[feats].sample(min(20000,len(F)),random_state=0)
    err=np.max(np.abs(NumpyGBM(dump).predict_proba(sample.values)[:,1]-m.predict_proba(sample)[:,1]))
    assert err < 1e-6, err
    meta=dict(features=feats,threshold=.15,pred_radius_m=.25,gsd=.1,
        loro_exact_f1=.72624,geographic_group_exact_f1=.607891,
        geographic_group_baseline_f1=.532940,regions=sorted(F.region.unique()),rows=len(F),
        feature_recipe="raw_plus_within_source_map_percentile_ranks")
    json.dump(meta,open(OUT/"finds.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
    print("rows",len(F),"features",len(feats),"numpy max error",err,"threshold",meta["threshold"])

if __name__=="__main__": main()
