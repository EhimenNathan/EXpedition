from pathlib import Path
import sys,geopandas as gpd,pandas as pd,numpy as np
ROOT=Path('/home/nathan/Expedition Competition');sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'Codex Work')]
from ae_layer_oof import layer_gt
from ae_det_oof import ae_folds
from sp_fusion_oof import fc
from koz3.labels import to_gt_geojson
from koz3.score import score
C='fortifikatsii';SET=[(n,m) for n in (0,5,10,20,30,50) for m in (.4,.5,.55,.6,.65,.7)]
def f1(v):
 tp,fp,fn=v;return 2*tp/max(2*tp+fn+.5*fp,1e-12)
RF=ae_folds();folds=sorted(set(RF.values()));L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=layer_gt(L[(L['mod']=='Ae')&L.region.isin(RF)])
D=pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_layers_cands.parquet');D=D[(D.cls==C)&(D.score>=.4)].copy()
GT={f:to_gt_geojson(L[L.region.map(RF)==f]) for f in folds};cnt={}
for st in SET:
 n,m=st;keep=[]
 for reg,q in D.groupby('region'):
  if len(q)>=n and q.score.max()>=m:keep.append(q)
 P=pd.concat(keep,ignore_index=True) if keep else D.iloc[:0]
 for f in folds:
  _,r=score(fc(P[P.region.map(RF)==f]),GT[f],classes=[C]);v=r[C];cnt[(st,f)]=np.array((v['tp'],v['fp'],v['fn']),int)
total=np.zeros(3,int);chosen={}
for f in folds:
 st=max(SET,key=lambda s:f1(sum((cnt[(s,g)] for g in folds if g!=f),np.zeros(3,int))))
 chosen[f]=st;total+=cnt[(st,f)]
base=sum((cnt[((0,.4),f)] for f in folds),np.zeros(3,int))
print('base',f1(base),tuple(base));print('nested',f1(total),tuple(total),'chosen',chosen)
for f in folds:print('outer',f,chosen[f],f1(cnt[(chosen[f],f)]),tuple(cnt[(chosen[f],f)]))
pooled=[]
for st in SET:
 v=sum((cnt[(st,f)] for f in folds),np.zeros(3,int));pooled.append((f1(v),st,tuple(v)))
for x in sorted(pooled,reverse=True)[:15]:print('pooled',x)
