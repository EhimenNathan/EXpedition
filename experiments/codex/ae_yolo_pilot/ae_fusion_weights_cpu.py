"""CPU diagnostic of three predeclared Ae FCOS/D-FINE fusion weights."""
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
from koz3.sp_fuse import fuse
from sp_fusion_oof import fc

THR=(.18,.22,.26,.30,.34)
W=(.5,1.,2.) # D-FINE relative to FCOS

def main():
    sources=[]
    for file in ('oof_ae_fcos_fast_cands.parquet','oof_ae_dfj_cands.parquet'):
        d=pd.read_parquet(ROOT/'work/ae'/file)
        sources.append(d[d.score>=.2][['region','cls','score','x','y']])
    site_fold={r['region']:r['fold'] for r in json.loads((ROOT/'data/ae_prep/ann.json').read_text())}
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='Ae')&L.region.isin(site_fold)]
    G={fold:to_gt_geojson(L[L.region.map(site_fold)==fold]) for fold in range(4)}
    results={}
    for w in W:
        d=fuse(sources,np.array([1.,w]),match_m=4.)
        d['fold']=d.region.map(site_fold)
        for fold in range(4):
            x=d[d.fold==fold]
            for t in THR:
                s,_=score(fc(x[x.score>=t]),G[fold]);results[(w,fold,t)]=s
        nested=[]
        for fold in range(4):
            choice=max(THR,key=lambda t:np.mean([results[(w,k,t)] for k in range(4) if k!=fold]))
            nested.append(results[(w,fold,choice)])
        print('D-FINE weight',w,'nested mean',round(float(np.mean(nested)),6),
              'folds',np.round(nested,5).tolist(),flush=True)

if __name__=='__main__':main()
