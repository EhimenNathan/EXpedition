#!/usr/bin/env python3
"""Exact one-fold screen of P2 rescue jointly with the region kurgan gate.

The specialist can override the conservative base-ensemble gate when it has a
strong independent response.  Existing predictions are otherwise unchanged.
This is a screen; any winner must be rerun on all folds with nested selection.
"""
import argparse,sys
from pathlib import Path
import geopandas as gpd,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_assemble_oof import models,region_folds
from sp_dual_oof import fuse_cls,K
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry

def main():
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int,required=True);a.add_argument('--path');q=a.parse_args()
 path=Path(q.path) if q.path else Path(__file__).parent/f'fold{q.fold}/oof_candidates.parquet'
 Y=pd.read_parquet(path);N=('dfine','rtdetr','hm','dfnodn','balq','balq3');M=models(extra=list(N[3:]));rf=region_folds()
 regs=sorted({r for r,f in rf.items() if f==q.fold});L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=L[(L['mod']=='SpOr')&L.region.isin(regs)]
 S=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet');S=S[S.region.isin(regs)];geom=scene_geometry()
 F=fuse_cls([M[n][M[n].region.isin(regs)] for n in N]);F=F[F.score>=np.where(F.cls=='arkhitektury',.15,.25)].reset_index(drop=True)
 keep=[]
 for r in F.itertuples():
  d=min([(np.hypot(r.x-cx,r.y-cy)/max(h,1),sm) for cx,cy,h,sm in geom.get(r.region,[])] or [(0.,True)])
  keep.append(d[1] or d[0]<=.5 or r.score>=.35)
 F=F[np.asarray(keep,bool)].reset_index(drop=True);kk=F[F.cls.isin(K)].groupby('region').score.max();weak=set(kk[kk<.45].index)
 ex=[(r.region,K[1] if r.cls==K[0] else K[0],r.score*.99,r.x,r.y) for r in F[F.cls.isin(K)].itertuples()
     if (r.s_kt if r.cls==K[0] else r.s_kp)>=.4]
 B=pd.concat([F[['region','cls','score','x','y']],pd.DataFrame(ex,columns=['region','cls','score','x','y'])],ignore_index=True)
 Y=Y[Y.region.isin(regs)];Y=fuse([Y[['region','cls','score','x','y']]],[1.],match_m=8.)
 gt=to_gt_geojson(L)
 def run(t,override,gate_on):
  p=Y[Y.score>=t].copy();pm=p.groupby('region').score.max();rescued=set(pm[pm>=override].index)
  b=B if not gate_on else B[~(B.cls.isin(K)&B.region.isin(weak-rescued))]
  # Independent candidates only; avoid duplicate emissions around an existing kurgan.
  out=[];bk=b[b.cls.isin(K)]
  for r in p.itertuples():
   z=bk[bk.region==r.region]
   if len(z) and np.hypot(z.x-r.x,z.y-r.y).min()<8:continue
   if gate_on and r.region in weak and r.region not in rescued:continue
   out.append((r.region,K[0],min(.99,float(r.score)),r.x,r.y))
  g=pd.concat([b,pd.DataFrame(out,columns=['region','cls','score','x','y'])],ignore_index=True)
  return score(fc(gate(g,S)),gt)[0],len(out),len(rescued),len(weak)
 print('base',run(2.,2.,False),'gate',run(2.,2.,True),'weak regions',len(weak))
 for t in (.02,.03,.05,.08,.10,.15,.20,.30,.40,.50):
  print('t',t,'ungated',run(t,2.,False),'gated',run(t,2.,True),'override@.25',run(t,.25,True),'override@.5',run(t,.5,True))

if __name__=='__main__':main()
