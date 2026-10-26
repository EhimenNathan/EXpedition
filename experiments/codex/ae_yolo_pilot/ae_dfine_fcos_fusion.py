"""CPU-only equal fusion of independent Ae D-FINE and FCOS geographic OOF.

The score threshold is nested: each held-out fold uses the value selected on
the other three folds. Reports full Ae official-default score and damaged F1.
"""
import json
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
HERE=ROOT/'Codex Work/ae_yolo_pilot'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_fusion_oof import fc

THRESHOLDS=(.18,.22,.26,.30,.34)

def main():
    sources=[]
    for file in ('oof_ae_fcos_fast_cands.parquet','oof_ae_dfj_cands.parquet'):
        d=pd.read_parquet(ROOT/'work/ae'/file)
        sources.append(d[d.score>=.2][['region','cls','score','x','y']])
    mixed=fuse(sources,np.ones(2),match_m=4.0)
    site_fold={r['region']:r['fold'] for r in json.loads((ROOT/'data/ae_prep/ann.json').read_text())}
    mixed['fold']=mixed.region.map(site_fold)
    assert mixed.fold.notna().all()
    out=HERE/'fusion_pilot';out.mkdir(exist_ok=True)
    mixed.to_parquet(out/'equal_candidates.parquet')
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='Ae')&L.region.isin(site_fold)]
    gt={fold:to_gt_geojson(L[L.region.map(site_fold)==fold]) for fold in range(4)}
    scores={}
    for fold in range(4):
        d=mixed[mixed.fold==fold]
        for t in THRESHOLDS:
            overall,res=score(fc(d[d.score>=t]),gt[fold])
            scores[(fold,t)]=(overall,res['kurgany_povrezhdennye']['f1'])
            print('fold',fold,'threshold',t,'AeF1',round(overall,6),
                  'damagedF1',round(res['kurgany_povrezhdennye']['f1'],6),
                  'n',int((d.score>=t).sum()),flush=True)
    nested=[]
    for fold in range(4):
        train={t:np.mean([scores[(k,t)][0] for k in range(4) if k!=fold]) for t in THRESHOLDS}
        chosen=max(train,key=train.get)
        nested.append(scores[(fold,chosen)][0])
        print('nested heldout',fold,'chosen',chosen,'AeF1',round(nested[-1],6))
    print('nested mean fold AeF1',round(float(np.mean(nested)),6))

if __name__=='__main__':main()
