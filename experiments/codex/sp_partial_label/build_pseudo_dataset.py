#!/usr/bin/env python3
"""Build satellite training labels with high-confidence OOF pseudo-kurgans in regions lacking kurgan annotation."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from shapely.geometry import Point
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sp_assemble_oof import models
from sp_dual_oof import fuse_cls,K
src=ROOT/'kaggle_import/code/work/sp'; out=ROOT/'work/sp_pseudo45'; out.mkdir(parents=True,exist_ok=True)
for n in ('scenes.csv','img'):
 p=out/n
 if not p.exists(): p.symlink_to(src/n, target_is_directory=(n=='img'))
sc=pd.read_csv(src/'scenes.csv'); lb=pd.read_parquet(src/'labels.parquet')
annot=set(lb.loc[lb.cls.isin(('курганы_целые','курганы_поврежденные')),'site'])
N=('dfine','rtdetr','hm','dfnodn','balq','balq3'); M=models(extra=list(N[3:])); F=fuse_cls([M[n] for n in N])
F=F[F.cls.isin(K)&(F.score>=.45)&~F.region.isin(annot)].copy()
F['cls']=F.cls.map({'kurgany_povrezhdennye':'курганы_поврежденные','kurgany_tselye':'курганы_целые'})
rows=[]
for q in F.itertuples():
 for r in sc[sc.site==q.region].itertuples():
  tr=Transformer.from_crs('EPSG:3857',r.crs,always_xy=True); x,y=tr.transform(q.x,q.y); T=Affine(*json.loads(r.T)); c,rr=~T*(x,y)
  if 0<=c<r.W and 0<=rr<r.H:
   # 5 m circle is conservative and gives a stable center target.
   rows.append(dict(site=r.site,scene=r.scene,cls=q.cls,wkb=Point(x,y).buffer(5,12).wkb,score=q.score))
add=pd.DataFrame(rows)
if len(add):
 # Remove near duplicates within each scene/class using highest score.
 keep=[]
 for _,g in add.sort_values('score',ascending=False).groupby(['scene','cls']):
  pts=[]
  for i,z in g.iterrows():
   p=Point(__import__('shapely').wkb.loads(z.wkb).centroid)
   if all(p.distance(t)>8 for t in pts): keep.append(i);pts.append(p)
 add=add.loc[keep]
merged=pd.concat([lb,add[['site','scene','cls','wkb']]],ignore_index=True)
merged.to_parquet(out/'labels.parquet')
print('base',len(lb),'pseudo fused',len(F),'assigned',len(add),'regions',add.site.nunique() if len(add) else 0,'merged',len(merged))
print(add.groupby('cls').size().to_dict() if len(add) else {})
