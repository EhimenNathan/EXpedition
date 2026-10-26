#!/usr/bin/env python3
"""Geographic-group outer validation for the MagGr self-normalized finds model.

The 13 nominal regions are only three archaeological/geographic families.  This
audit holds each whole family out and chooses the probability threshold through
inner leave-one-region-out predictions from the remaining families.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from shapely.geometry import Point
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_import/code/work/geo/finds_v3"
sys.path.insert(0, str(ROOT / "kaggle_import/code"))
from koz3.official.fastmatch import soft_match, prf

DROP = {"x", "y", "tgt", "uid", "region", "render"}
THR = [.01, .02, .03, .05, .075, .10, .15, .20, .25, .30, .40, .50, .60]


def family(region):
    n = int(region.split("_", 1)[0])
    return "sudbishchi" if n == 200 else ("semibratnee" if 201 <= n <= 207 else "gnezdovo")


def normalize(df):
    out = df.copy(); cols = [c for c in out if c not in DROP and np.issubdtype(out[c].dtype, np.number)]
    for _, idx in out.groupby("uid", sort=False).groups.items():
        x = out.loc[idx, cols]
        q = x.rank(method="average", pct=True); q.columns = ["rank__" + c for c in cols]
        med = x.median(); scale = (x.quantile(.75) - x.quantile(.25)).replace(0, 1.)
        z = ((x-med)/scale).clip(-8, 8); z.columns = ["rz__" + c for c in cols]
        out.loc[idx, q.columns] = q.to_numpy(np.float32)
        out.loc[idx, z.columns] = z.to_numpy(np.float32)
    return out


def clf():
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=.025, num_leaves=25,
        min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.65,
        reg_lambda=3., random_state=0, n_jobs=8, verbose=-1)


def counts(frame, gt, pcol, thresholds=THR):
    out = {}
    for uid, g in frame.groupby("uid"):
        gts = [Point(x,y) for x,y in gt[uid]]
        for t in thresholds:
            s=g[g[pcol]>=t]
            out[uid,t]=np.asarray(soft_match([(Point(x,y).buffer(.25,8),p) for x,y,p in zip(s.x,s.y,s[pcol])],gts)[:3],float)
    return out


def main():
    raw = pd.concat([pd.read_parquet(p) for p in sorted((SRC/"units").glob("*__orig.parquet"))],ignore_index=True)
    raw = raw.drop_duplicates(["uid","x","y"]).reset_index(drop=True)
    d=normalize(raw); d["family"]=d.region.map(family)
    base=[c for c in raw if c not in DROP]
    variants={"raw":base,"raw_rank":base+[c for c in d if c.startswith("rank__")],
              "raw_rank_rz":base+[c for c in d if c.startswith(("rank__","rz__"))]}
    gt={k:np.asarray(v) for k,v in json.load(open(SRC/"gt.json",encoding="utf-8")).items()}
    rows=[]
    for fam in sorted(d.family.unique()):
      train=d[d.family!=fam]; test=d[d.family==fam]
      for name,feats in variants.items():
        # Inner region-OOF predictions for threshold selection, restricted to outer training families.
        inn=train[["uid","region","x","y","tgt"]].copy(); inn["p"]=np.nan
        for r in sorted(train.region.unique()):
            te=train.region.eq(r); tr=~te
            inn.loc[te,"p"]=clf().fit(train.loc[tr,feats],train.loc[tr,"tgt"]).predict_proba(train.loc[te,feats])[:,1]
        cc=counts(inn,gt,"p")
        uids=inn.uid.unique()
        t=max(THR,key=lambda q:prf(*np.sum([cc[u,q] for u in uids],axis=0))[2])
        p=clf().fit(train[feats],train.tgt).predict_proba(test[feats])[:,1]
        te=test[["uid","region","x","y","tgt"]].copy();te["p"]=p
        ct=counts(te,gt,"p",[t]); c=np.sum([ct[u,t] for u in te.uid.unique()],axis=0)
        P,R,F=prf(*c)
        rows.append(dict(heldout_family=fam,variant=name,threshold=t,P=P,R=R,F1=F,tp=c[0],fp=c[1],fn=c[2]))
        print(fam,name,"thr",t,"P/R/F",round(P,4),round(R,4),round(F,4),flush=True)
    out=pd.DataFrame(rows)
    # aggregate counts across the three untouched families
    agg=[]
    for name,g in out.groupby("variant"):
        c=g[["tp","fp","fn"]].sum().to_numpy(); P,R,F=prf(*c)
        agg.append(dict(variant=name,P=P,R=R,F1=F,tp=c[0],fp=c[1],fn=c[2],min_family_f1=g.F1.min()))
    out.to_csv(Path(__file__).with_suffix(".folds.csv"),index=False)
    pd.DataFrame(agg).to_csv(Path(__file__).with_suffix(".summary.csv"),index=False)
    print(pd.DataFrame(agg).to_string(index=False))

if __name__=="__main__": main()
