"""Site-disjoint context model, with variant and cutoff fixed on folds 0-2."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

HERE=Path(__file__).resolve().parent

def predict(F,cols,train,test):
    imp=SimpleImputer(strategy='median').fit(F.loc[train,cols])
    Xtr=imp.transform(F.loc[train,cols]);Xte=imp.transform(F.loc[test,cols])
    scale=StandardScaler().fit(Xtr)
    model=LogisticRegression(C=.1,max_iter=1000,class_weight='balanced').fit(scale.transform(Xtr),F.loc[train,'has_k'])
    return model.predict_proba(scale.transform(Xte))[:,1]

def main():
    B=pd.read_parquet(HERE/'sp_region_presence_probe.parquet')
    C=pd.read_parquet(HERE/'sp_scene_context_features.parquet')
    F=B.merge(C,on='region',how='left',validate='one_to_one')
    base=[c for c in B if c not in ('fold','region','has_k','gt_count','p_site')]
    scene=[c for c in C if c!='region']
    variants={'base':base,'scene':scene,'combined':base+scene}
    inner={}
    for name,cols in variants.items():
        F['p_'+name]=np.nan
        for fold in (0,1,2,3):
            tr=F.fold.isin(set((0,1,2))-{fold}) if fold<3 else F.fold.isin((0,1,2))
            te=F.fold.eq(fold)
            F.loc[te,'p_'+name]=predict(F,cols,tr,te)
        q=F[F.fold<3]
        inner[name]=roc_auc_score(q.has_k,q['p_'+name])
        print(name,'inner_auc',round(inner[name],4),
              'outer_auc',round(roc_auc_score(F[F.fold==3].has_k,F.loc[F.fold==3,'p_'+name]),4),
              'outer_ap',round(average_precision_score(F[F.fold==3].has_k,F.loc[F.fold==3,'p_'+name]),4),flush=True)
    chosen=max(variants,key=lambda k:inner[k]);col='p_'+chosen
    cutoff=float(F[(F.fold<3)&(F.has_k==1)][col].min())-1e-6
    F['drop']=F[col]<cutoff
    print('chosen on inner',chosen,'cutoff',cutoff,flush=True)
    for fold in range(4):
        q=F[F.fold==fold];d=q[q['drop']]
        print('fold',fold,'sites dropped',len(d),'zero',int((d.has_k==0).sum()),
              'positive',int((d.has_k==1).sum()),'GT lost',int(d.gt_count.sum()),flush=True)
    F.to_parquet(HERE/'sp_scene_context_probe.parquet',index=False)
    (HERE/'sp_scene_context_probe.json').write_text(json.dumps({'inner_auc':inner,'chosen':chosen,'cutoff':cutoff},indent=2))

if __name__=='__main__':main()
