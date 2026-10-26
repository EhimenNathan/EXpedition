#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,geopandas as gpd,pandas as pd
R=Path('/home/nathan/Expedition Competition');sys.path[:0]=[str(R),str(R/'scripts'),str(R/'Codex Work')]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc
KP='kurgany_povrezhdennye'
ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);a=ap.parse_args()
d=pd.read_parquet(R/f'Codex Work/ae_yolo_p2/fold{a.fold}/oof_layers.parquet');L=gpd.read_parquet(R/'data/labels_3857.parquet');L=L[(L['mod']=='Ae')&L.region.isin(d.region.unique())];gt=to_gt_geojson(layer_gt(L))
for mode in ('native','damaged'):
 print(mode)
 for t in (.03,.05,.08,.1,.15,.2,.25,.3,.4,.5):
  q=d[d.score>=t].copy()
  if mode=='damaged':q['cls']=KP
  s,r=score(fc(q),gt);v=r[KP];print(t,round(s,6),len(q),'KP',round(v['f1'],6),v['tp'],v['fp'],v['fn'])
