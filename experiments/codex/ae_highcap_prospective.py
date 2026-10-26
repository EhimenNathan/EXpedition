#!/usr/bin/env python3
"""Prospective high-cap FCOS audit: select on fold 2, open fold 0 once."""
from pathlib import Path
import sys, geopandas as gpd, pandas as pd, numpy as np
R=Path('/home/nathan/Expedition Competition');sys.path[:0]=[str(R),str(R/'scripts'),str(R/'Codex Work')]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc
KP='kurgany_povrezhdennye'
def run(d,gt,t,cap):
 q=d[d.score>=t].sort_values('score',ascending=False).groupby('region',group_keys=False).head(cap).copy();q['cls']=KP
 _,r=score(fc(q),gt,classes=[KP]);v=r[KP];return v['f1'],(v['tp'],v['fp'],v['fn']),len(q)
def main():
 L=gpd.read_parquet(R/'data/labels_3857.parquet');L=L[L['mod']=='Ae']
 ds={2:pd.read_parquet(R/'Codex Work/ae_cap_f2_c1000.parquet'),0:pd.read_parquet(R/'Codex Work/ae_cap_f0_c5000_s0.08.parquet')}
 G={f:to_gt_geojson(layer_gt(L[L.region.isin(ds[f].region.unique())])) for f in ds}
 opts=[]
 for t in (.2,.25,.3,.35,.4,.45,.5,.55,.6):
  for cap in (20,50,100,200,400,800):opts.append((run(ds[2],G[2],t,cap)[0],t,cap))
 opts.sort(reverse=True);print('fold2 top');
 for z in opts[:15]:print(z,run(ds[2],G[2],z[1],z[2]))
 _,t,cap=opts[0];print('COMMITTED',t,cap);print('fold0',run(ds[0],G[0],t,cap))
 # baseline existing frozen FCOS layer candidates under identical policy
 b=pd.read_parquet(R/'work/ae/oof_ae_fcos_fast_layers_cands.parquet')
 for f in (2,0):print('baseline',f,run(b[b.fold==f],G[f],t,cap))
if __name__=='__main__':main()
