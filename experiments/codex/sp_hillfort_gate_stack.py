"""Cross-fit a hillfort scene gate from three independent OOF scores."""
import os
import numpy as np, pandas as pd
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")

def main():
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 from sklearn.pipeline import make_pipeline
 from sklearn.metrics import roc_auc_score
 S=pd.read_parquet(os.path.join(ROOT,"work/spor/oof_scene_cls.parquet"))
 S["tif"]=[f[len(r)+2:].rsplit("__",1)[0]+".tif" for r,f in zip(S.region,S.file)]
 D=pd.read_csv(os.path.join(ROOT,"work/hf_dinov3/loc_points_oof_1024only.csv"))
 D=S[["region","tif","fold","y","p"]].merge(D[["region","tif","map_max","p_scene"]],on=["region","tif"],validate="one_to_one")
 X=np.c_[D.p,D.map_max,D.p_scene,D.p*D.map_max,D.p*D.p_scene,D.map_max*D.p_scene]
 y=D.y.values.astype(int); f=D.fold.values; D["p_stack"]=0.
 for fold in sorted(np.unique(f)):
  tr=f!=fold; te=f==fold
  m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=3000)).fit(X[tr],y[tr])
  D.loc[te,"p_stack"]=m.predict_proba(X[te])[:,1]
 print("scenes",len(D),"positives",int(y.sum()))
 for c in ("p","map_max","p_scene","p_stack"):
  print(c,"AUC",round(roc_auc_score(y,D[c]),4),"fold",[round(roc_auc_score(y[f==q],D.loc[f==q,c]),3) for q in sorted(np.unique(f))])
 out=os.path.join(ROOT,"work/hf_dinov3/hillfort_gate_stack_oof.csv"); D.to_csv(out,index=False); print("saved",out)

if __name__=="__main__":main()
