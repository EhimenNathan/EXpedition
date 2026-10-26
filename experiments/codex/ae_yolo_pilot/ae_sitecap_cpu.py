"""Nested geographic CPU audit of Ae FCOS cross-class hedge plus site cap.

Fold 0 remains untouched for the independent Kaggle pilot. Site caps are
applied by confidence within each predicted class and site, not by labels.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'Codex Work/ae_yolo_pilot')]
from ae_crossclass_cpu import append_hedge,INTACT,DAMAGED,f1
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

CAPS=(None,20,40,80,120,200)


def main():
    D=pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet')
    D=D[D.fold.isin([1,2,3])]
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    F=D[['region','fold']].drop_duplicates().set_index('region').fold.to_dict()
    L=L[(L['mod']=='Ae')&L.region.isin(F)]
    G={fold:to_gt_geojson(L[L.region.map(F)==fold]) for fold in [1,2,3]}
    counts={}
    for fold in [1,2,3]:
        d=D[D.fold==fold]
        base=d[d.score>=.4]
        hedge=pd.concat([append_hedge(d,DAMAGED,INTACT,.4),
                         append_hedge(d,INTACT,DAMAGED,.3)],ignore_index=True)
        for mode,x in [('base',base),('hedge',pd.concat([base,hedge],ignore_index=True))]:
            for cap in CAPS:
                y=x if cap is None else (x.sort_values('score',ascending=False)
                    .groupby(['region','cls'],sort=False).head(cap))
                _,res=score(fc(y),G[fold],classes=[INTACT,DAMAGED])
                counts[(mode,cap,fold)]={c:(res[c]['tp'],res[c]['fp'],res[c]['fn'])
                                         for c in (INTACT,DAMAGED)}
    for c in (INTACT,DAMAGED):
        print('\n',c)
        total=np.zeros(3,int)
        for fold in [1,2,3]:
            candidates={}
            for mode in ('base','hedge'):
                for cap in CAPS:
                    pooled=np.sum([counts[(mode,cap,k)][c] for k in [1,2,3] if k!=fold],axis=0)
                    candidates[(mode,cap)]=f1(*pooled)
            choice=max(candidates,key=candidates.get)
            val=np.array(counts[(*choice,fold)][c]);total+=val
            print('heldout',fold,'choice',choice,'counts',val.tolist(),'f1',round(f1(*val),5))
        baseline=np.sum([counts[('base',None,k)][c] for k in [1,2,3]],axis=0)
        print('nested',round(f1(*total),5),total.tolist(),
              'baseline',round(f1(*baseline),5),baseline.tolist())
        for mode in ('base','hedge'):
            for cap in CAPS:
                pooled=np.sum([counts[(mode,cap,k)][c] for k in [1,2,3]],axis=0)
                print(mode,cap,round(f1(*pooled),5),pooled.tolist())


if __name__=='__main__':main()
