"""Cross-fit a conservative selector for DINO hillfort peaks after rank 1.

Positive means the candidate is the first (highest map score) peak inside a
hillfort polygon not already hit by rank 1.  This mirrors one-to-one matching
and makes all duplicate peaks in the same polygon negatives.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")
sys.path[:0]=[ROOT,os.path.join(ROOT,"scripts")]

def main():
 import geopandas as gpd
 from shapely.geometry import Point
 from sklearn.linear_model import LogisticRegression
 from sklearn.ensemble import HistGradientBoostingClassifier
 from sklearn.metrics import roc_auc_score
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sp_rescore_oof import scene_geometry

 p=os.path.join(ROOT,"work/hf_dinov3/loc_multipeak_oof.csv"); D=pd.read_csv(p)
 # p_scene is an honest DINO scene-presence probability available for all 175
 # scenes, unlike the deployed gate cache which covers only the exact pipeline's
 # common OOF region subset.
 S=pd.read_csv(os.path.join(ROOT,"work/hf_dinov3/loc_points_oof.csv"))
 D=D.merge(S[["region","tif","p_scene"]],on=["region","tif"],how="left",validate="many_to_one")
 assert D.p_scene.notna().all()
 geom=scene_geometry(); half=[]
 for r in D.itertuples():
  gg=geom.get(r.region,[]); half.append(min(gg,key=lambda z:np.hypot(r.cx3857-z[0],r.cy3857-z[1]))[2] if gg else 300.)
 D["half"]=half
 top=D[D["rank"]==1][["region","tif","x3857","y3857"]].rename(columns={"x3857":"x1","y3857":"y1"})
 D=D.merge(top,on=["region","tif"],validate="many_to_one")
 D["d1"]=np.hypot(D.x3857-D.x1,D.y3857-D.y1)/np.maximum(D.half,1)
 D["dc"]=np.hypot(D.x3857-D.cx3857,D.y3857-D.cy3857)/np.maximum(D.half,1)

 L=gpd.read_parquet(os.path.join(ROOT,"data/labels_3857.parquet")); L=L[(L["mod"]=="SpOr")&(L.class_name=="gorodishcha")]
 polys={r:list(g.geometry) for r,g in L.groupby("region")}
 D["y"]=0
 for (reg,tif),ix in D.groupby(["region","tif"]).groups.items():
  q=D.loc[ix].sort_values("rank"); pp=polys.get(reg,[])
  if not pp: continue
  first=Point(float(q.iloc[0].x3857),float(q.iloc[0].y3857))
  uncovered=[j for j,g in enumerate(pp) if not g.contains(first)]
  for j in uncovered:
   cand=q[(q["rank"]>=2)&q.apply(lambda r: pp[j].contains(Point(float(r.x3857),float(r.y3857))),axis=1)]
   if len(cand): D.loc[cand.index[0],"y"]=1
 E=D[D["rank"]>=2].copy()
 feats=["map_score","ratio","rank","p_scene","d1","dc","half"]
 X=E[feats].replace([np.inf,-np.inf],np.nan).fillna(0).values; y=E.y.values; f=E.fold.values
 E["meta_logit"]=np.nan; E["meta_hgb"]=np.nan
 for fold in sorted(np.unique(f)):
  tr=f!=fold; te=f==fold
  lr=make_pipeline(StandardScaler(),LogisticRegression(C=.2,class_weight="balanced",max_iter=3000)).fit(X[tr],y[tr])
  hg=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,l2_regularization=3,min_samples_leaf=20,random_state=0)
  sw=np.where(y[tr], max(1,(~y[tr].astype(bool)).sum()/max(1,y[tr].sum())), 1.)
  hg.fit(X[tr],y[tr],sample_weight=sw)
  E.loc[E.index[te],"meta_logit"]=lr.predict_proba(X[te])[:,1]
  E.loc[E.index[te],"meta_hgb"]=hg.predict_proba(X[te])[:,1]
 print("extra candidates",len(E),"positives",int(y.sum()))
 for c in ("map_score","ratio","meta_logit","meta_hgb"):
  print(c,"AUC",round(roc_auc_score(y,E[c]),4))
 for c in ("meta_logit","meta_hgb"):
  print("\n",c)
  for t in (.2,.3,.4,.5,.6,.7,.8,.9):
   z=E[E[c]>=t]; print(t,"kept",len(z),"tp candidates",int(z.y.sum()),"precision",round(z.y.mean(),3) if len(z) else 0)
 out=os.path.join(ROOT,"work/hf_dinov3/loc_multipeak_meta_oof.csv"); E.to_csv(out,index=False); print("saved",out)

if __name__=="__main__": main()
