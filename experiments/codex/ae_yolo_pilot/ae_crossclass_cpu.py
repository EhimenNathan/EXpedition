"""Low-resource Ae kurgan cross-class hedge on existing FCOS OOF.

Fold 0 stays untouched for the independent Kaggle pilot. Presets are chosen
on the other two folds and scored on the third of folds 1/2/3. This is a
diagnostic, not a deployable model or a claim about public validation.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path('/home/nathan/Expedition Competition')
HERE=ROOT/'Codex Work/ae_yolo_pilot'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score,WEIGHTS
from sp_fusion_oof import fc

INTACT='kurgany_tselye'
DAMAGED='kurgany_povrezhdennye'
THRESH=(.3,.4,.5,.6)


def f1(tp,fp,fn):
    return 2*tp/max(2*tp+fn+.5*fp,1e-9) if tp+fn else 0.


def append_hedge(D,source,target,threshold,base_threshold=.4,no_target_within_m=4.):
    base=D[(D.cls==target)&(D.score>=base_threshold)]
    source_rows=D[(D.cls==source)&(D.score>=threshold)]
    out=[]
    for region,rows in source_rows.groupby('region'):
        existing=base[base.region==region]
        xy=rows[['x','y']].to_numpy()
        if len(existing):
            distance,_=cKDTree(existing[['x','y']].to_numpy()).query(xy)
            rows=rows.loc[distance>no_target_within_m]
        if len(rows):
            h=rows.copy()
            h['cls']=target
            h['score']=h.score*.9
            out.append(h)
    return pd.concat(out,ignore_index=True) if out else D.iloc[:0].copy()


def main():
    D=pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet')
    D=D[(D.fold.isin([1,2,3]))&D.cls.isin([INTACT,DAMAGED])]
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    F=D[['region','fold']].drop_duplicates().set_index('region').fold.to_dict()
    L=L[(L['mod']=='Ae')&L.region.isin(F)]
    G={fold:to_gt_geojson(L[L.region.map(F)==fold]) for fold in [1,2,3]}
    counts={}
    for target in [INTACT,DAMAGED]:
        source=DAMAGED if target==INTACT else INTACT
        for t in (None,*THRESH):
            for fold in [1,2,3]:
                d=D[D.fold==fold]
                base=d[(d.cls==target)&(d.score>=.4)]
                if t is None:
                    pred=base
                else:
                    hedge=append_hedge(d,source,target,t)
                    pred=pd.concat([base,hedge],ignore_index=True)
                _,res=score(fc(pred),G[fold],classes=[target])
                x=res[target]
                counts[(target,t,fold)]=(x['tp'],x['fp'],x['fn'])
    for target in [INTACT,DAMAGED]:
        print('\n',target)
        nested=np.zeros(3,int)
        for fold in [1,2,3]:
            options={t:np.sum([counts[(target,t,k)] for k in [1,2,3] if k!=fold],axis=0)
                     for t in (None,*THRESH)}
            choice=max(options,key=lambda t:f1(*options[t]))
            val=np.array(counts[(target,choice,fold)])
            nested+=val
            print('heldout',fold,'chosen',choice,'counts',val.tolist(),'f1',round(f1(*val),5))
        baseline=np.sum([counts[(target,None,k)] for k in [1,2,3]],axis=0)
        print('nested',round(f1(*nested),5),nested.tolist(),
              'baseline',round(f1(*baseline),5),baseline.tolist())
        for t in THRESH:
            pooled=np.sum([counts[(target,t,k)] for k in [1,2,3]],axis=0)
            print('fixed',t,round(f1(*pooled),5),pooled.tolist())


if __name__=='__main__':
    main()
