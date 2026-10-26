"""Fold-safe kurgan-site presence probe using frozen DINOv2 scene embeddings.

Embeddings were computed without mound labels. Each site's scene vectors are
averaged. Standardization, PCA, and logistic fitting use only training folds.
Sites without cached embeddings retain the existing score-based probability.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.scene_cls import scenes

def main():
    S=scenes();X=np.load(ROOT/'work/spor/dinov2b_scene_feats.npy',mmap_mode='r')
    assert len(S)==len(X)
    sites=sorted({s['region'] for s in S})
    V=np.stack([np.asarray(X[[i for i,s in enumerate(S) if s['region']==r]],np.float32).mean(0) for r in sites])
    E=pd.DataFrame({'region':sites,'emb_idx':np.arange(len(sites))})
    F=pd.read_parquet(HERE/'sp_region_presence_probe.parquet').merge(E,on='region',how='left')
    F['p_dino']=np.nan;F['p_dino_score']=np.nan
    score_cols=[c for c in F if c not in ('fold','region','has_k','gt_count','p_site','emb_idx','p_dino','p_dino_score')]
    for fold in (0,1,2,3):
        tr=(F.fold.isin(set((0,1,2))-{fold}) if fold<3 else F.fold.isin((0,1,2))) & F.emb_idx.notna()
        te=F.fold.eq(fold)&F.emb_idx.notna()
        U=V[F.loc[tr,'emb_idx'].astype(int)];W=V[F.loc[te,'emb_idx'].astype(int)]
        scale=StandardScaler().fit(U);U=scale.transform(U);W=scale.transform(W)
        pca=PCA(n_components=min(16,len(U)-2),svd_solver='randomized',random_state=20261025).fit(U)
        U=pca.transform(U);W=pca.transform(W)
        y=F.loc[tr,'has_k']
        m=LogisticRegression(C=.01,max_iter=1000,class_weight='balanced').fit(U,y)
        F.loc[te,'p_dino']=m.predict_proba(W)[:,1]
        mu=F.loc[tr,score_cols].mean().to_numpy();sd=F.loc[tr,score_cols].std().to_numpy()+1e-6
        a=(F.loc[tr,score_cols].to_numpy()-mu)/sd;b=(F.loc[te,score_cols].to_numpy()-mu)/sd
        m2=LogisticRegression(C=.01,max_iter=1000,class_weight='balanced').fit(np.column_stack([U,a]),y)
        F.loc[te,'p_dino_score']=m2.predict_proba(np.column_stack([W,b]))[:,1]
        z=F[te]
        if z.has_k.nunique()==2:
            print('fold',fold,'sites',len(z),'positives',int(z.has_k.sum()),
                  'base_auc',round(roc_auc_score(z.has_k,z.p_site),4),
                  'dino_auc',round(roc_auc_score(z.has_k,z.p_dino),4),
                  'combined_auc',round(roc_auc_score(z.has_k,z.p_dino_score),4),flush=True)
    for subset,name in [(F[F.fold<3],'inner'),(F[F.fold==3],'outer')]:
        q=subset[subset.emb_idx.notna()]
        print(name,'n',len(q),'positives',int(q.has_k.sum()),flush=True)
        for col in ('p_site','p_dino','p_dino_score'):
            print(name,col,'auc',round(roc_auc_score(q.has_k,q[col]),4),
                  'ap',round(average_precision_score(q.has_k,q[col]),4),flush=True)
    F.to_parquet(HERE/'sp_dino_site_context.parquet',index=False)

if __name__=='__main__':main()
