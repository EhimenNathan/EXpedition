"""Nested geographic audit of fixed Ae fusion density caps (CPU only)."""
import json
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

THR=(.18,.22,.26)
CAP=(None,20,50,100,200)

def main():
    D=pd.read_parquet(ROOT/'Codex Work/ae_yolo_pilot/fusion_pilot/equal_candidates.parquet')
    site_fold={r['region']:r['fold'] for r in json.loads((ROOT/'data/ae_prep/ann.json').read_text())}
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='Ae')&L.region.isin(site_fold)]
    G={fold:to_gt_geojson(L[L.region.map(site_fold)==fold]) for fold in range(4)}
    results={}
    for fold in range(4):
        d=D[D.fold==fold]
        for t in THR:
            x=d[d.score>=t]
            for cap in CAP:
                y=x if cap is None else x.sort_values('score',ascending=False).groupby(['region','cls'],sort=False).head(cap)
                s,res=score(fc(y),G[fold]);results[(fold,t,cap)]=(s,res)
    nested=[]
    for fold in range(4):
        options={(t,c):np.mean([results[(k,t,c)][0] for k in range(4) if k!=fold]) for t in THR for c in CAP}
        choice=max(options,key=options.get)
        s,res=results[(fold,*choice)]
        nested.append(s)
        print('heldout',fold,'choice',choice,'AeF1',round(s,6),
              'damagedF1',round(res['kurgany_povrezhdennye']['f1'],6))
    print('nested mean',round(float(np.mean(nested)),6),'folds',np.round(nested,5).tolist())
    for t in THR:
        for cap in CAP:
            a=[results[(f,t,cap)][0] for f in range(4)]
            print('fixed',t,cap,'mean',round(float(np.mean(a)),6),'folds',np.round(a,5).tolist())

if __name__=='__main__':main()
