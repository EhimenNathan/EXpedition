"""Conservative prospective fold-3 addition of dense Ae points.

The fixed FCOS+joint baseline is preserved byte-for-byte.  Dense points at
the precommitted 0.20 threshold are added only when no baseline point of the
same class lies within four true metres.
"""
import argparse,json,sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path('/home/nathan/Expedition Competition');CW=ROOT/'Codex Work'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(CW/'ae_yolo_pilot')]
from ae_scale_fusion_probe import model_input,make,evaluate
from ae_crossclass_cpu import DAMAGED,INTACT
from koz3.labels import to_gt_geojson

ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,default=3);ap.add_argument('--min-dense',type=int,default=0);ap.add_argument('--pred',default='');ap.add_argument('--threshold',type=float,default=.2);a=ap.parse_args()
ann=json.loads((ROOT/'data/ae_prep/ann.json').read_text());sites={r['region'] for r in ann if r['fold']==a.fold}
sources=[model_input(ROOT/'work/ae'/f,sites,.2) for f in ('oof_ae_fcos_fast_cands.parquet','oof_ae_dfj_cands.parquet')]
base=make(sources)
pred=Path(a.pred) if a.pred else CW/f'ae_dense_f{a.fold}_cands.parquet'
d=pd.read_parquet(pred);d=d[(d.score>=a.threshold)&d.region.isin(sites)]
if a.min_dense:
 dense_sites=set(d[d.cls==DAMAGED].groupby('region').size().loc[lambda x:x>=a.min_dense].index)
 d=d[d.region.isin(dense_sites)].copy()
 print('density gate',a.min_dense,'sites',sorted(dense_sites),flush=True)
labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet');labels=labels[(labels['mod']=='Ae')&labels.region.isin(sites)];gt=to_gt_geojson(labels)
for mode in ('damaged','both'):
 add=d[d.cls==DAMAGED].copy() if mode=='damaged' else d.copy();keep=[]
 for r,g in add.groupby('region'):
  for c,h in g.groupby('cls'):
   b=base[(base.region==r)&(base.cls==c)];tree=cKDTree(b[['x','y']]) if len(b) else None
   k=1/np.cos(np.radians(np.degrees(2*np.arctan(np.exp(h.y.mean()/6378137))-np.pi/2)))
   keep.extend(h.index if tree is None else h.index[tree.query(h[['x','y']],k=1)[0]>4*k])
 aug=pd.concat([base,add.loc[keep,base.columns]],ignore_index=True).sort_values('score',ascending=False).groupby(['region','cls'],sort=False).head(100)
 print(mode,'base',evaluate(base,gt),'aug',evaluate(aug,gt),flush=True)
 if mode=='damaged':
  for site in sorted(sites):
   sg=to_gt_geojson(labels[labels.region==site]);before=evaluate(base[base.region==site],sg);after=evaluate(aug[aug.region==site],sg)
   nd=int(((d.region==site)&(d.cls==DAMAGED)).sum());nb=int(((base.region==site)&(base.cls==DAMAGED)).sum())
   print('site',site,'dense20',nd,'base_n',nb,'delta',round(after['ae_f1']-before['ae_f1'],5),
         'Dtp',after['classes'][DAMAGED]['tp']-before['classes'][DAMAGED]['tp'],
         'Dfp',after['classes'][DAMAGED]['fp']-before['classes'][DAMAGED]['fp'],flush=True)
