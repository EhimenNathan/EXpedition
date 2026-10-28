"""Fuse the whole-scene DINOv3 hillfort localizer into the exact v21 satellite pipeline.

Tests localization alone with the deployed scene gate, both old/new points, and
replacement by the independent DINOv3 presence score.  Every choice is selected
on the other geographic folds and compared with exact v21 decoding.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
sys.path[:0]=[ROOT,os.path.join(ROOT,"scripts")]

def main(points="work/hf_dinov3/loc_points_oof.csv"):
 import geopandas as gpd
 from koz3.labels import to_gt_geojson
 from koz3.nested import fold_counts,nested,weighted,_add_all
 from robust_accept import site_counts,report
 from sp_assemble_oof import models,region_folds
 from sp_dual_oof import fuse_cls,K
 from sp_fusion_oof import fc
 from sp_rescore_oof import scene_geometry
 names=("dfine","rtdetr","hm","dfnodn","balq","balq3")
 M=models(extra=list(names[3:])); regions=sorted(set.intersection(*[set(M[n].region) for n in names]))
 RF=region_folds(); folds=sorted({RF[r] for r in regions})
 L=gpd.read_parquet(os.path.join(ROOT,"data/labels_3857.parquet")); L=L[(L["mod"]=="SpOr")&L.region.isin(regions)]
 S=pd.read_parquet(os.path.join(ROOT,"work/spor/oof_scene_cls.parquet")); S=S[S.region.isin(regions)].copy()
 D3=pd.read_csv(os.path.join(ROOT,points)); D3=D3[D3.region.isin(regions)].copy()
 MP_PATH=os.path.join(ROOT,"work/hf_dinov3/loc_multipeak_oof.csv")
 MP=pd.read_csv(MP_PATH) if os.path.exists(MP_PATH) else None
 if MP is not None: MP=MP[MP.region.isin(regions)].copy()
 S["tif"]=[f[len(r)+2:].rsplit("__",1)[0]+".tif" for r,f in zip(S.region,S.file)]
 D3=D3.rename(columns={"x3857":"d3_x","y3857":"d3_y"})
 S=S.merge(D3[["region","tif","d3_x","d3_y","map_max","p_scene"]],on=["region","tif"],how="left",validate="one_to_one")
 assert S[["d3_x","d3_y","map_max","p_scene"]].notna().all().all(),"scene/localizer join failed"
 S["d3"]=(S.map_max+S.p_scene)/2
 geom=scene_geometry(); M={n:M[n][M[n].region.isin(regions)] for n in names}
 Fsets={0.0:fuse_cls([M[n] for n in names]).reset_index(drop=True)}
 extra=os.environ.get("SP_EXTRA_MEMBER")
 if extra:
  from sp_add_member_oof import fused
  new=pd.read_parquet(os.path.join(ROOT,f"work/spor/oof_det_{extra}_cands.parquet"))[["region","cls","score","x","y"]]
  Fsets[.5]=fused(M,new[new.region.isin(regions)],.5).reset_index(drop=True)
 half={}
 for r in S.itertuples():
  g=geom.get(r.region,[]); half[r.Index]=min(g,key=lambda z:np.hypot(r.x-z[0],r.y3857-z[1]))[2] if g else 300.

 def base_candidates(regs,use_extra=False):
  F0=Fsets[.5 if use_extra else 0.0]
  F=F0[F0.region.isin(regs)].copy(); F=F[F.score>=np.where(F.cls=="arkhitektury",.15,.25)]
  keep=[]
  for r in F.itertuples():
   d=min([(np.hypot(r.x-cx,r.y-cy)/max(h,1),sm) for cx,cy,h,sm in geom.get(r.region,[])] or [(0.,True)])
   keep.append(d[1] or d[0]<=.5 or r.score>=.35)
  F=F[np.array(keep,bool)]
  ex=[(r.region,K[1] if r.cls==K[0] else K[0],r.score*.99,r.x,r.y) for r in F[F.cls.isin(K)].itertuples()
      if (r.s_kt if r.cls==K[0] else r.s_kp)>=.4]
  return pd.concat([F[["region","cls","score","x","y"]],pd.DataFrame(ex,columns=["region","cls","score","x","y"])],ignore_index=True)

 def pipeline(st,regs):
  mode,g=st; use_extra=mode=="loc_extra"; F0=Fsets[.5 if use_extra else 0.0]
  GC=F0[(F0.cls=="gorodishcha")&(F0.score>=.02)]
  G=base_candidates(regs,use_extra); X=S[S.region.isin(regs)]
  if mode in ("base","loc","loc_extra","both","rescue","multipeak"):
   G=G[~((G.cls=="gorodishcha")&G.region.isin(set(X[X.p<.1].region)))]
   A=X[X.p>=.1]
  else:
   A=X[X.d3>=g]
   good=set(A.region); G=G[~((G.cls=="gorodishcha")&~G.region.isin(good))]
  rows=[]
  for r in A.itertuples():
   cx,cy=r.x,r.y3857
   if mode in ("base","both"):
    c=GC[(GC.region==r.region)&(GC.score>=.02)]; c=c[np.hypot(c.x-cx,c.y-cy)<=.2*half[r.Index]]
    if len(c): b=c.loc[c.score.idxmax()]; cx,cy=b.x,b.y
    rows.append((r.region,"gorodishcha",r.p,cx,cy))
   if mode in ("loc","loc_extra","both","rescue","multipeak"):
    rows.append((r.region,"gorodishcha",r.p*(.99 if mode=="both" else 1),r.d3_x,r.d3_y))
   if mode=="multipeak" and MP is not None:
    q=MP[(MP.region==r.region)&(MP.tif==r.tif)&(MP["rank"]>=2)&(MP.ratio>=g)]
    for z in q.itertuples():
     # Keep matching order below the established scene-gate point.
     rows.append((r.region,"gorodishcha",r.p*(.99**int(z.rank)),z.x3857,z.y3857))
   if mode=="d3": rows.append((r.region,"gorodishcha",r.d3,r.d3_x,r.d3_y))
  if mode=="rescue":
   for r in X[(X.p<.1)&(X.d3>=g)].itertuples(): rows.append((r.region,"gorodishcha",r.d3,r.d3_x,r.d3_y))
  return pd.concat([G,pd.DataFrame(rows,columns=G.columns)],ignore_index=True)

 SS=[("base",0.),("loc",0.),("both",0.)]+([("loc_extra",0.)] if extra else [])+([("multipeak",g) for g in (.5,.6,.7,.8,.9)] if MP is not None else [])+[("rescue",g) for g in (.6,.7,.8,.9)]+[("d3",g) for g in (.1,.2,.3,.4,.5,.6,.7)]
 gt={f:to_gt_geojson(L[L.region.map(RF)==f]) for f in folds}
 C=fold_counts(lambda st,f:fc(pipeline(st,[r for r in regions if RF[r]==f])),SS,folds,gt)
 for st in SS:
  tot=_add_all(C[st].values()); pf=' '.join(f'{weighted(C[st][f]):.4f}' for f in folds)
  print(f"{st}: pooled {weighted(tot):.4f} folds [{pf}] gor {tuple(int(v) for v in tot.get('gorodishcha',(0,0,0)))}")
 h,ch,o,os_,_=nested(C); base=weighted(_add_all(C[("base",0.)].values()))
 print(f"v21 {base:.4f} | nested {h:.4f} ({h-base:+.4f}) oracle {o:.4f} @ {os_} chosen {ch}")
 for f in folds: print(f" fold {f}: {weighted(C[('base',0.)][f]):.4f} -> {weighted(C[ch[f]][f]):.4f}")
 fixed=max(SS,key=lambda s:weighted(_add_all(C[s].values())))
 gtr={r:to_gt_geojson(L[L.region==r]) for r in regions}
 robust_settings=tuple(dict.fromkeys((("base",0.),("loc",0.),fixed)))
 CS=site_counts(lambda st,r:fc(pipeline(st,[r])),robust_settings,regions,gtr)
 report("v21 DINOv3 hillfort location-only",CS[("base",0.)],CS[("loc",0.)])
 report(f"v21 DINOv3 hillfort {fixed}",CS[("base",0.)],CS[fixed])

if __name__=="__main__": main(*sys.argv[1:])
