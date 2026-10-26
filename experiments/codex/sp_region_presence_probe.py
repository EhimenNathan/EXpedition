"""Geographic outer-fold test of label-free satellite mound-site evidence.

Region labels are derived from training GT only. Folds 0-2 train a small
regularized classifier; fold 3 is an untouched domain-shift test. This is a
diagnostic, not a deployed gate.
"""
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
K=("kurgany_povrezhdennye","kurgany_tselye")

def main():
    C=pd.read_parquet(HERE/"sp_ring_features.parquet")
    L=gpd.read_parquet(ROOT/"data/labels_3857.parquet")
    L=L[(L['mod']=='SpOr')&L.class_name.isin(K)]
    labels=L.groupby('region').size().to_dict()
    rows=[]
    for (fold,region),g in C.groupby(['fold','region']):
        s=np.sort(g.score.to_numpy())
        if not len(s):continue
        d=dict(fold=int(fold),region=region,has_k=int(region in labels),gt_count=int(labels.get(region,0)),
               log_n=np.log1p(len(s)),max_score=float(s[-1]),top5=float(s[-min(5,len(s)):].mean()),
               q95=float(np.quantile(s,.95)),q75=float(np.quantile(s,.75)),
               class_frac=float((g.cls==K[0]).mean()))
        for t in (.15,.25,.35,.45,.55,.65):d[f'log_n_{int(t*100)}']=np.log1p(int((s>=t).sum()))
        rows.append(d)
    F=pd.DataFrame(rows)
    cols=[c for c in F if c not in ('fold','region','has_k','gt_count')]
    F['p_site']=np.nan
    for fold in range(4):
        tr=F.fold.isin(set((0,1,2))-{fold}) if fold<3 else F.fold.isin((0,1,2))
        te=F.fold.eq(fold)
        scale=StandardScaler().fit(F.loc[tr,cols]);xtr=scale.transform(F.loc[tr,cols]);xte=scale.transform(F.loc[te,cols])
        model=LogisticRegression(C=.1,max_iter=1000,class_weight='balanced').fit(xtr,F.loc[tr,'has_k'])
        F.loc[te,'p_site']=model.predict_proba(xte)[:,1]
        y=F.loc[te,'has_k']
        print('fold',fold,'sites',int(te.sum()),'positive',int(y.sum()),
              'max_auc',round(roc_auc_score(y,F.loc[te,'max_score']),4),
              'site_auc',round(roc_auc_score(y,F.loc[te,'p_site']),4),
              'site_ap',round(average_precision_score(y,F.loc[te,'p_site']),4),flush=True)
    F.to_parquet(HERE/'sp_region_presence_probe.parquet',index=False)
    for thr in (.05,.1,.15,.2,.3,.4):
        te=F[F.fold==3]; drop=te.p_site<thr
        print('outer gate',thr,'zero sites removed',int(((te.has_k==0)&drop).sum()),'of',int((te.has_k==0).sum()),
              'positive sites removed',int(((te.has_k==1)&drop).sum()),'of',int((te.has_k==1).sum()),
              'GT objects lost',int(te.loc[drop,'gt_count'].sum()),flush=True)

if __name__=='__main__':main()
