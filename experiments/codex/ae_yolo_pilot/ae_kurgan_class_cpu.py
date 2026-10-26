"""CPU-only cross-class kurgan check with geographic development/final split."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT=Path('/home/nathan/Expedition Competition')
HERE=ROOT/'Codex Work/ae_yolo_pilot'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

DAM='kurgany_povrezhdennye';INT='kurgany_tselye'

def main():
    D=pd.read_parquet(HERE/'radial_pilot/features.parquet')
    D=D[D.cls.isin((DAM,INT))].copy().reset_index(drop=True)
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    site_fold={r['region']:r['fold'] for r in json.loads((ROOT/'data/ae_prep/ann.json').read_text())}
    L=L[(L['mod']=='Ae')&L.region.isin(site_fold)]
    gt={(site,cls):list(g.geometry) for (site,cls),g in L.groupby(['region','class_name'])}
    actual=[]
    for row in D.itertuples():
        k=1/np.cos(np.radians(np.degrees(2*np.arctan(np.exp(row.y/6378137.))-np.pi/2)))
        p=Point(row.x,row.y)
        dist=[min((g.distance(p) for g in gt.get((row.region,c),[])),default=float('inf')) for c in (DAM,INT)]
        actual.append(1+int(dist[1]<dist[0]) if min(dist)<=4*k else 0)
    D['target']=actual
    print('candidate class labels',D.target.value_counts().to_dict(),flush=True)
    cols=['score']+[c for c in D if c.startswith('pixel_')]
    D['source_damaged']=(D.cls==DAM).astype(np.int8);cols.append('source_damaged')
    D['prob_dam']=np.nan;D['prob_int']=np.nan
    for hold in range(4):
        folds=({1,2,3}-{hold}) if hold else {1,2,3}
        tr=D[D.fold.isin(folds)];va=D[D.fold==hold]
        model=lgb.LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=4,
            learning_rate=.035,min_child_samples=75,reg_lambda=7.,
            verbose=-1,n_jobs=4,random_state=20261024)
        model.fit(tr[cols],tr.target)
        prob=model.predict_proba(va[cols])
        D.loc[va.index,['prob_dam','prob_int']]=prob[:,1:3]
    assert D[['prob_dam','prob_int']].notna().all().all()
    G={fold:to_gt_geojson(L[L.region.map(site_fold)==fold]) for fold in range(4)}
    raw=pd.read_parquet(HERE/'fusion_pilot/equal_candidates.parquet')
    results={}
    for fold in range(4):
        base=raw[(raw.fold==fold)&(raw.score>=.18)].copy()
        d=D[D.fold==fold]
        for t in (None,.05,.1,.15,.2):
            if t is None:
                pred=base
            else:
                hedge=d[((d.cls==DAM)&(d.prob_int>=t))|
                        ((d.cls==INT)&(d.prob_dam>=t))].copy()
                src=hedge.cls.to_numpy()
                hedge['cls']=np.where(src==DAM,INT,DAM)
                hedge['score']=np.where(src==DAM,hedge.prob_int,hedge.prob_dam)*hedge.score
                pred=pd.concat([base,hedge[base.columns]],ignore_index=True)
            pred=pred.sort_values('score',ascending=False).groupby(['region','cls'],sort=False).head(100)
            s,res=score(fc(pred),G[fold]);results[(fold,t)]=(s,res[DAM]['f1'],res[INT]['f1'])
            print('fold',fold,'gate',t,'AeF1',round(s,5),'dam',round(res[DAM]['f1'],5),
                  'int',round(res[INT]['f1'],5),flush=True)
    for hold in range(4):
        folds=[f for f in (1,2,3) if f!=hold]
        choice=max((None,.05,.1,.15,.2),key=lambda t:np.mean([results[(f,t)][0] for f in folds]))
        print('selected',hold,'gate',choice,'score',results[(hold,choice)],flush=True)

if __name__=='__main__':main()
