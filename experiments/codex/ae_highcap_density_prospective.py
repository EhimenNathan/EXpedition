#!/usr/bin/env python3
"""High-cap FCOS cemetery-density rescue: tune fold 2, audit fold 0 once."""
from pathlib import Path
import sys, geopandas as gpd,pandas as pd,numpy as np
from scipy.spatial import cKDTree
R=Path('/home/nathan/Expedition Competition');sys.path[:0]=[str(R),str(R/'scripts'),str(R/'Codex Work')]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc
KP='kurgany_povrezhdennye'
def prep(d):
 d=d[d.score>=.2].copy();out=[]
 for reg,g in d.groupby('region'):
  g=g.copy();k=1/np.cos(np.radians(np.degrees(2*np.arctan(np.exp(g.y.mean()/6378137.))-np.pi/2)))
  t=cKDTree(g[['x','y']].to_numpy(float))
  for rad in (20.,40.,60.,100.,150.):g[f'n{int(rad)}']=[len(x) for x in t.query_ball_point(g[['x','y']].to_numpy(float),rad*k)]
  out.append(g)
 return pd.concat(out,ignore_index=True)
def run(d,gt,st):
 rad,need,alpha,cap=st;n=f'n{int(rad)}';q=d[d[n]>=need].copy();q['rank']=q.score*(np.minimum(q[n],50)**alpha);q=q.sort_values('rank',ascending=False).groupby('region',group_keys=False).head(cap);q['cls']=KP
 _,r=score(fc(q),gt,classes=[KP]);v=r[KP];return v['f1'],(v['tp'],v['fp'],v['fn']),len(q)
def main():
 L=gpd.read_parquet(R/'data/labels_3857.parquet');L=L[L['mod']=='Ae'];raw={2:pd.read_parquet(R/'Codex Work/ae_cap_f2_c1000.parquet'),0:pd.read_parquet(R/'Codex Work/ae_cap_f0_c5000_s0.08.parquet')};D={f:prep(x) for f,x in raw.items()};G={f:to_gt_geojson(layer_gt(L[L.region.isin(D[f].region.unique())])) for f in D}
 opts=[]
 for rad in (20.,40.,60.,100.,150.):
  for need in (1,2,3,5,8,12,20):
   for alpha in (0.,.25,.5,1.):
    for cap in (50,100,200,400,800):opts.append((run(D[2],G[2],(rad,need,alpha,cap))[0],(rad,need,alpha,cap)))
 opts.sort(reverse=True);print('fold2 top');
 for v,st in opts[:20]:print(v,st,run(D[2],G[2],st))
 st=opts[0][1];print('COMMITTED',st);print('fold0',run(D[0],G[0],st));print('no-density fold0',run(D[0],G[0],(20.,1,0.,400)))
if __name__=='__main__':main()
