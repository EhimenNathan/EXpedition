"""Nested region-presence test using other archaeological-class evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

HERE=Path(__file__).resolve().parent

def pred(F,cols,tr,te):
    imp=SimpleImputer(strategy='median').fit(F.loc[tr,cols])
    A=imp.transform(F.loc[tr,cols]);B=imp.transform(F.loc[te,cols])
    sc=StandardScaler().fit(A)
    m=LogisticRegression(C=.1,max_iter=1000,class_weight='balanced').fit(sc.transform(A),F.loc[tr,'has_k'])
    return m.predict_proba(sc.transform(B))[:,1]

def main():
    B=pd.read_parquet(HERE/'sp_region_presence_probe.parquet')
    X=pd.read_parquet(HERE/'sp_crossclass_site_features.parquet')
    F=B.merge(X,on='region',how='left',validate='one_to_one')
    base=[c for c in B if c not in ('fold','region','has_k','gt_count','p_site')]
    other=[c for c in X if c!='region']
    variants={'base':base,'crossclass':base+other}
    for name,cols in variants.items():
        F['p_'+name]=np.nan
        for fold in (0,1,2,3):
            tr=F.fold.isin(set((0,1,2))-{fold}) if fold<3 else F.fold.isin((0,1,2))
            te=F.fold.eq(fold)
            F.loc[te,'p_'+name]=pred(F,cols,tr,te)
        for part,q in [('inner',F[F.fold<3]),('outer',F[F.fold==3])]:
            print(name,part,'n',len(q),'auc',round(roc_auc_score(q.has_k,q['p_'+name]),4),
                  'ap',round(average_precision_score(q.has_k,q['p_'+name]),4),flush=True)
        for fold,g in F.groupby('fold'):
            print(name,'fold',fold,'auc',round(roc_auc_score(g.has_k,g['p_'+name]),4),flush=True)
    F.to_parquet(HERE/'sp_crossclass_site_probe.parquet',index=False)

if __name__=='__main__':main()
