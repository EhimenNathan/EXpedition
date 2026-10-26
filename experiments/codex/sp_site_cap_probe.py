"""Satellite false-positive hotspot audit: cap kurgan emissions per site.

Caps are picked on folds 0-2 and prospectively scored on fold 3 through the
same fused candidate, framing, class hedge, hillfort gate, and exact scorer.
"""
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(Path(__file__).resolve().parent)]
from koz3.labels import to_gt_geojson
from koz3.nested import weighted,_add_all
from koz3.score import score
from sp_assemble_oof import models,region_folds
from sp_dual_oof import fuse_cls
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry

K=('kurgany_povrezhdennye','kurgany_tselye')
NAMES=('dfine','rtdetr','hm','dfnodn','balq','balq3')

def main():
 M=models(extra=list(NAMES[3:]));regions=sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
 RF=region_folds();F=fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES])
 F=F[((F.cls.isin(K))&(F.score>=.25))|((F.cls=='arkhitektury')&(F.score>=.15))|
     ((~F.cls.isin(K+('arkhitektury',)))&(F.score>=.25))].copy()
 geom=scene_geometry();keep=[]
 for r in F.itertuples():
  d=min([(np.hypot(r.x-cx,r.y-cy)/max(h,1),sm) for cx,cy,h,sm in geom.get(r.region,[])] or [(0.,True)])
  keep.append(d[1] or d[0]<=.5 or r.score>=.35)
 F=F[np.array(keep)].copy()
 S=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet')
 L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=L[(L['mod']=='SpOr')&L.region.isin(regions)]
 GT={f:to_gt_geojson(L[L.region.map(RF)==f]) for f in range(4)}
 C={}
 for cap in (None,20,40,80):
  G=F.copy()
  if cap is not None:
   k=G[G.cls.isin(K)].sort_values('score',ascending=False).groupby('region',sort=False).head(cap)
   G=pd.concat([k,G[~G.cls.isin(K)]],ignore_index=True)
  extra=[]
  for r in G[G.cls.isin(K)].itertuples():
   other=K[1] if r.cls==K[0] else K[0]
   if (r.s_kt if other==K[1] else r.s_kp)>=.4:
    extra.append((r.region,other,r.score*.99,r.x,r.y))
  G=G[['region','cls','score','x','y']]
  if extra:G=pd.concat([G,pd.DataFrame(extra,columns=G.columns)],ignore_index=True)
  for f in range(4):
   pred=fc(gate(G[G.region.map(RF)==f],S[S.region.map(RF)==f]))
   _,by=score(pred,GT[f]);C[(cap,f)]={k:(v['tp'],v['fp'],v['fn']) for k,v in by.items()}
 for cap in (None,20,40,80):
  inn=weighted(_add_all(C[(cap,f)] for f in range(3)))
  out=weighted(C[(cap,3)])
  print('cap',cap,'inner',round(inn,6),'fold3',round(out,6),
        'folds',[round(weighted(C[(cap,f)]),5) for f in range(4)],flush=True)
 chosen=max((None,20,40,80),key=lambda c:weighted(_add_all(C[(c,f)] for f in range(3))))
 print('inner_chosen',chosen,'outer_delta',weighted(C[(chosen,3)])-weighted(C[(None,3)]),flush=True)

if __name__=='__main__':main()
