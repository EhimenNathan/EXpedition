"""Fit deployable all-training DINO scene and hillfort meta heads.

The evaluation remains the geographic cross-fit written by
sp_hillfort_gate_stack.py.  This script only fits the final deployment form
after that policy has been fixed; it does not report an in-sample score as
evidence.
"""
import os
import numpy as np
import pandas as pd

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")

def main(src="models/hf_dino_loc.npz",out="models/hf_dino_loc_stack.npz"):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 W=dict(np.load(os.path.join(ROOT,src)))
 Z=np.load(os.path.join(ROOT,"work/hf_dinov3/tokens1024.npz"),allow_pickle=True)
 tok=Z["tok"].astype(np.float32);sid=Z["sid"]
 X=np.stack([np.r_[tok[sid==s].mean(0),tok[sid==s].std(0)] for s in range(len(Z["meta"]))])
 y=np.array([Z["y"][sid==s].any() for s in range(len(Z["meta"]))],int)
 pca=PCA(64,random_state=0).fit(X);A=pca.transform(X);sc=StandardScaler().fit(A)
 lr=LogisticRegression(C=.05,class_weight="balanced",max_iter=3000).fit(sc.transform(A),y)
 W.update(scene_pca_mean=pca.mean_.astype(np.float32),scene_pca_comp=pca.components_.astype(np.float32),
          scene_mu=sc.mean_.astype(np.float32),scene_sd=sc.scale_.astype(np.float32),
          scene_coef=lr.coef_[0].astype(np.float32),scene_b=np.float32(lr.intercept_[0]))

 S=pd.read_parquet(os.path.join(ROOT,"work/spor/oof_scene_cls.parquet"))
 S["tif"]=[f[len(r)+2:].rsplit("__",1)[0]+".tif" for r,f in zip(S.region,S.file)]
 D=pd.read_csv(os.path.join(ROOT,"work/hf_dinov3/loc_points_oof_1024only.csv"))
 D=S[["region","tif","y","p"]].merge(D[["region","tif","map_max","p_scene"]],on=["region","tif"],validate="one_to_one")
 p,m,s=D.p.values,D.map_max.values,D.p_scene.values
 A=np.c_[p,m,s,p*m,p*s,m*s];sc2=StandardScaler().fit(A)
 lr2=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000).fit(sc2.transform(A),D.y.values.astype(int))
 W.update(stack_mu=sc2.mean_.astype(np.float32),stack_sd=sc2.scale_.astype(np.float32),
          stack_coef=lr2.coef_[0].astype(np.float32),stack_b=np.float32(lr2.intercept_[0]))
 path=os.path.join(ROOT,out);np.savez(path,**W)
 print("saved",path,"bytes",os.path.getsize(path),"scene",len(X),"stack",len(A))

if __name__=="__main__":main()
