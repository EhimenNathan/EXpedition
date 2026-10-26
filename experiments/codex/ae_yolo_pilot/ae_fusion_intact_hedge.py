"""Nested CPU test of duplicating select fused damaged candidates as intact."""
import json
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'Codex Work/ae_yolo_pilot')]
from ae_crossclass_cpu import append_hedge,INTACT,DAMAGED
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

OPTIONS=(None,.2,.25,.3)

def main():
    raw=pd.read_parquet(ROOT/'Codex Work/ae_yolo_pilot/fusion_pilot/equal_candidates.parquet')
    site_fold={r['region']:r['fold'] for r in json.loads((ROOT/'data/ae_prep/ann.json').read_text())}
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='Ae')&L.region.isin(site_fold)]
    G={fold:to_gt_geojson(L[L.region.map(site_fold)==fold]) for fold in range(4)}
    results={}
    for fold in range(4):
        d=raw[raw.fold==fold]
        base=d[d.score>=.18]
        for t in OPTIONS:
            pred=base if t is None else pd.concat([
                base,append_hedge(d,DAMAGED,INTACT,t,base_threshold=.18)],ignore_index=True)
            pred=pred.sort_values('score',ascending=False).groupby(['region','cls'],sort=False).head(100)
            s,res=score(fc(pred),G[fold]);results[(fold,t)]=(s,res)
            print('fixed fold',fold,'hedge',t,'AeF1',round(s,6),
                  'intactF1',round(res[INTACT]['f1'],6),'damagedF1',round(res[DAMAGED]['f1'],6),flush=True)
    nested=[]
    for fold in range(4):
        choice=max(OPTIONS,key=lambda t:np.mean([results[(k,t)][0] for k in range(4) if k!=fold]))
        s,res=results[(fold,choice)];nested.append(s)
        print('nested fold',fold,'choice',choice,'AeF1',round(s,6))
    print('nested mean',round(float(np.mean(nested)),6),'folds',np.round(nested,5).tolist())

if __name__=='__main__':main()
