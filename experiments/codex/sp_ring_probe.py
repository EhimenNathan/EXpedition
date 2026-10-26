"""Predefined leave-geographic-fold-out ring-feature objectness probe."""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

HERE=Path(__file__).resolve().parent

def fit_predict(F,cols,tr,te):
    d=F[tr]; y=d.y_obj.to_numpy()
    site_n=d.groupby("region").size().to_dict()
    w=np.array([1/np.sqrt(site_n[r]) for r in d.region])
    w*=np.where(y==1,(len(y)-y.sum())/max(1,y.sum()),1)
    m=lgb.LGBMClassifier(n_estimators=220,learning_rate=.05,num_leaves=15,max_depth=4,
                          min_child_samples=60,reg_lambda=5,colsample_bytree=.8,
                          subsample=.8,subsample_freq=1,verbosity=-1,n_jobs=2,
                          random_state=20261025)
    m.fit(d[cols],y,sample_weight=w)
    return m.predict_proba(F.loc[te,cols])[:,1]

def main():
    R=pd.read_parquet(HERE/"sp_ring_features.parquet")
    M=pd.read_parquet(HERE/"sp_morph_features.parquet")
    keys=["region","cls","score","x","y","fold","y_obj"]
    F=R.merge(M,on=keys,how="inner",validate="one_to_one")
    rings=[c for c in R if c not in keys]
    old=[c for c in M if c not in keys and c not in ("s_kp","s_kt")]
    core=["score","s_kp","s_kt"]
    F["p_ring"]=np.nan;F["p_all"]=np.nan;F["p_score"]=np.nan
    for fold in (0,1,2,3):
        tr=F.fold.isin(set((0,1,2))-{fold}) if fold<3 else F.fold.isin((0,1,2))
        te=F.fold.eq(fold)
        for name,cols in [("p_ring",rings+core),("p_all",rings+old+core),("p_score",core)]:
            F.loc[te,name]=fit_predict(F,cols,tr,te)
        y=F.loc[te,"y_obj"]
        print("fold",fold,"n",int(te.sum()),"positive",int(y.sum()),
              "raw",round(average_precision_score(y,F.loc[te,"score"]),4),
              "score",round(average_precision_score(y,F.loc[te,"p_score"]),4),
              "ring",round(average_precision_score(y,F.loc[te,"p_ring"]),4),
              "all",round(average_precision_score(y,F.loc[te,"p_all"]),4),
              "ring_auc",round(roc_auc_score(y,F.loc[te,"p_ring"]),4),flush=True)
    F[keys+["p_ring","p_all","p_score"]].to_parquet(HERE/"sp_ring_probe.parquet",index=False)

if __name__=="__main__":main()
