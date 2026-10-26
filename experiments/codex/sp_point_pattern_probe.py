"""Site-level kurgan intent from candidate point-process geometry.

All point-pattern features are label-free. Model fitting is geographic-fold
cross-fitted; fold 3 is a prospective test after inner-fold comparisons.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score,average_precision_score
from lightgbm import LGBMClassifier

HERE=Path(__file__).resolve().parent
P=pd.read_parquet(HERE/'sp_region_presence_probe.parquet')
F=pd.read_parquet(HERE/'sp_morph_features.parquet')

rows=[]
for region,g in F.groupby('region'):
    xy=g[['x','y']].to_numpy()
    if len(xy)<4:continue
    # EPSG:3857 metric distances shrink locally by cos(latitude).
    lat=np.arctan(np.sinh(float(np.median(xy[:,1]))/6378137.0))
    xy=np.unique(np.round(xy*np.cos(lat)*2)/2,axis=0)
    if len(xy)<4:continue
    tree=cKDTree(xy);dist,ind=tree.query(xy,k=min(7,len(xy)))
    d1=dist[:,1];d2=dist[:,2]
    vec=xy[ind[:,1]]-xy
    ang=np.arctan2(vec[:,1],vec[:,0])
    feat={'region':region,'nn_med':np.median(d1),'nn_cv':np.std(d1)/(np.mean(d1)+1e-6),
          'nn_q90':np.quantile(d1,.9),'nn_second_ratio':np.median(d2/(d1+1e-6)),
          'axis_order':abs(np.mean(np.exp(2j*ang)))}
    for radius in (5,10,20,40):
        count=np.array([len(z)-1 for z in tree.query_ball_point(xy,radius)])
        feat[f'count{radius}_med']=np.median(count)
        feat[f'count{radius}_cv']=np.std(count)/(np.mean(count)+1e-6)
        feat[f'count{radius}_q90']=np.quantile(count,.9)
    rows.append(feat)
G=P.merge(pd.DataFrame(rows),on='region',how='left')
base=[c for c in P if c not in ('region','fold','has_k','gt_count','p_site')]
pattern=[c for c in G if c not in P]
for variant,cols in [('base',base),('pattern',pattern),('combined',base+pattern)]:
    G['p_'+variant]=np.nan
    for fold in range(4):
        tr=G.fold.isin(tuple({0,1,2}-{fold}) if fold<3 else (0,1,2))
        te=G.fold.eq(fold)
        m=LGBMClassifier(n_estimators=70,num_leaves=4,min_child_samples=8,max_depth=3,
                         learning_rate=.04,reg_lambda=8,verbosity=-1,random_state=20261025,n_jobs=1)
        m.fit(G.loc[tr,cols],G.loc[tr,'has_k'])
        G.loc[te,'p_'+variant]=m.predict_proba(G.loc[te,cols])[:,1]
for name,g in [('inner',G[G.fold<3]),('outer',G[G.fold==3])]:
    for c in ('p_site','p_base','p_pattern','p_combined'):
        print(name,c,'auc',round(roc_auc_score(g.has_k,g[c]),4),'ap',round(average_precision_score(g.has_k,g[c]),4),flush=True)
for fold,g in G.groupby('fold'):
    print('fold',fold,'n',len(g),'k',int(g.has_k.sum()),'base',round(roc_auc_score(g.has_k,g.p_base),3),
          'combined',round(roc_auc_score(g.has_k,g.p_combined),3),flush=True)
G.to_parquet(HERE/'sp_point_pattern_probe.parquet',index=False)
