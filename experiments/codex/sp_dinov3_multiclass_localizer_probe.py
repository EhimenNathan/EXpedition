#!/usr/bin/env python3
"""Site-held-out DINOv3-SAT token localizer screen for every satellite class.

The accepted v26 path uses this mechanism only for hillforts. This prospective
probe asks whether the same frozen whole-scene representation contains unused
location/presence signal for architecture, fortification, and kurgans. PCA and
logistic heads are fit only on the other geographic folds.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
CLASSES=('kurgany_povrezhdennye','kurgany_tselye','gorodishcha','arkhitektury','fortifikatsii')

def main(path='work/hf_dinov3/tokens.npz'):
 import geopandas as gpd, pandas as pd
 import shapely
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from sklearn.preprocessing import StandardScaler
 from scipy.ndimage import gaussian_filter
 from koz3.sp_prep import site_folds
 Z=np.load(ROOT/path,allow_pickle=True);T=Z['tok'].astype(np.float32);pos=Z['pos'];sid=Z['sid'];meta=Z['meta'];grid=Z['grid']
 L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=L[L['mod']=='SpOr'];fo=site_folds(L)
 regions=np.array([m[0] for m in meta]);sf=np.array([fo[r] for r in regions]);tf=sf[sid]
 Y={}
 for cls in CLASSES:
  y=np.zeros(len(T),bool)
  for r,G in L[L.class_name==cls].groupby('region'):
   ss=np.flatnonzero(regions==r)
   if not len(ss):continue
   ii=np.flatnonzero(np.isin(sid,ss));u=G.geometry.union_all()
   y[ii]=shapely.contains_xy(u,pos[ii,2],pos[ii,3])
  Y[cls]=y
  print(cls,'positive tokens',int(y.sum()),'positive scenes',sum(y[sid==s].any() for s in range(len(meta))),flush=True)
 P={c:np.zeros(len(T),np.float32) for c in CLASSES}
 for f in sorted(set(sf)):
  tr,te=tf!=f,tf==f
  pca=PCA(64,random_state=20261029).fit(T[tr][::4]);A=pca.transform(T);sc=StandardScaler().fit(A[tr]);A=sc.transform(A)
  for cls in CLASSES:
   y=Y[cls]
   if y[tr].sum()==0:continue
   m=LogisticRegression(C=.02,class_weight='balanced',max_iter=2000).fit(A[tr],y[tr])
   P[cls][te]=m.predict_proba(A[te])[:,1]
  print('fold',f,'done',flush=True)
 rows=[]
 for cls in CLASSES:
  y=Y[cls];has=[];hit=[];mx=[];lx=[];ly=[];cx=[];cy=[]
  for s,(gh,gw) in enumerate(grid):
   ii=np.flatnonzero(sid==s);h=bool(y[ii].any());g=gaussian_filter(P[cls][ii].reshape(gh,gw),.5,mode='nearest');a=int(g.argmax())
   has.append(h);hit.append(bool(y[ii][a]));mx.append(float(g.max()));lx.append(float(pos[ii][a,2]));ly.append(float(pos[ii][a,3]))
   cx.append(float(pos[ii,2].mean()));cy.append(float(pos[ii,3].mean()))
  has=np.array(has);hit=np.array(hit);mx=np.array(mx)
  auc=roc_auc_score(has,mx) if 0<has.sum()<len(has) else float('nan')
  print(f'{cls:24s} token AUC {roc_auc_score(y,P[cls]):.3f} presence AUC {auc:.3f} localizer hit {hit[has].mean() if has.any() else 0:.3f} ({hit[has].sum()}/{has.sum()})')
  for s in range(len(meta)):rows.append(dict(cls=cls,region=regions[s],tif=meta[s][1],fold=sf[s],has=has[s],hit=hit[s],map_max=mx[s],x3857=lx[s],y3857=ly[s],cx3857=cx[s],cy3857=cy[s]))
 pd.DataFrame(rows).to_parquet(ROOT/'work/sp_dinov3_multiclass_localizer_probe.parquet',index=False)

if __name__=='__main__':main(*sys.argv[1:])
