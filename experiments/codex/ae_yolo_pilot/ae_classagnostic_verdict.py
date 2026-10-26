"""Diagnostic upper bound: class-agnostic kurgan location recall on Ae OOF.

All kurgan predictions are temporarily assigned to each target class in turn.
This is for error attribution only; the reassigned predictions are not shipped.
"""
import argparse
import json
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

K=('kurgany_povrezhdennye','kurgany_tselye')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred',required=True)
    ap.add_argument('--threshold',type=float,default=.4)
    ap.add_argument('--folds',default='0,1,2,3')
    a=ap.parse_args()
    wanted={int(x) for x in a.folds.split(',')}
    raw=pd.read_parquet(a.pred)
    raw=raw[raw.fold.isin(wanted)]
    manifest=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    F={r['region']:r['fold'] for r in manifest if r['fold'] in wanted}
    assert set(raw.region).issubset(F)
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='Ae')&L.region.isin(F)]
    for fold in sorted(wanted):
        d=raw[(raw.fold==fold)&(raw.score>=a.threshold)]
        gt=to_gt_geojson(L[L.region.map(F)==fold])
        for cls in K:
            true=d[d.cls==cls]
            _,b=score(fc(true),gt,classes=[cls])
            loc=d[d.cls.isin(K)].copy();loc['cls']=cls
            _,u=score(fc(loc),gt,classes=[cls])
            print('fold',fold,cls,'true_class_recall',round(b[cls]['r'],5),
                  'any_kurgan_recall_upper',round(u[cls]['r'],5),
                  'GT',b[cls]['tp']+b[cls]['fn'],flush=True)

if __name__=='__main__':main()
