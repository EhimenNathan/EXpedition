"""Site-disjoint EO-Swin verifier and exact Ae reranking evaluation."""
from __future__ import annotations
import sys
from pathlib import Path
import geopandas as gpd, numpy as np, pandas as pd
from scipy.special import expit
from shapely import points
from shapely.strtree import STRtree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path('/home/nathan/Expedition Competition');CW=ROOT/'Codex Work'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(CW)]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc
KP='kurgany_povrezhdennye'

def label(d,gt):
    y=np.zeros(len(d),np.uint8);best=np.zeros(len(d),np.uint8)
    for region,idx in d.groupby('region').groups.items():
        ii=np.asarray(list(idx),int);g=gt[(gt.region==region)&(gt.class_name==KP)]
        if not len(g):continue
        pair=STRtree(g.geometry.to_numpy()).query(points(d.x.iloc[ii],d.y.iloc[ii]),predicate='within')
        if pair.shape[1]==0:continue
        ci=ii[pair[0]];y[ci]=1
        # Exactly one positive training example per reference object: the highest
        # scoring proposal that can match it. This mirrors greedy metric utility.
        z=pd.DataFrame({'ci':ci,'gi':pair[1],'s':d.score.iloc[ci].to_numpy()})
        pick=z.sort_values('s').groupby('gi').tail(1).ci.to_numpy();best[pick]=1
    return y,best

def f1(v):
    tp,fp,fn=v;return 2*tp/max(2*tp+fn+.5*fp,1e-12)

def main():
    d=pd.read_parquet(CW/'ae_sheet_selfadapt_features.parquet').reset_index(drop=True)
    E=np.load(CW/'ae_swin_eo_embeddings.npy').astype(np.float32);E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-8
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=L[(L['mod']=='Ae')&L.region.isin(d.region.unique())]
    G=layer_gt(L).reset_index(drop=True);y,ybest=label(d,G);d['y_any']=y;d['y_best']=ybest
    print('labels any/best',int(y.sum()),int(ybest.sum()),'of',len(d),flush=True)
    hand=np.nan_to_num(d[['score','s_kp','s_kt','sim_seed','sim_knn']].to_numpy(np.float32),nan=0.0)
    X=np.concatenate([E,hand],axis=1);p=np.zeros(len(d),np.float32)
    for fold in sorted(d.fold.unique()):
        tr=d.fold!=fold;te=~tr
        sc=StandardScaler().fit(X[tr]);xt=sc.transform(X[tr]);xe=sc.transform(X[te])
        m=SGDClassifier(loss='log_loss',alpha=3e-4,class_weight='balanced',max_iter=2000,
                        tol=1e-5,random_state=100+int(fold),average=True)
        m.fit(xt,ybest[tr]);p[te]=m.predict_proba(xe)[:,1]
        print('fold',fold,'n',int(te.sum()),'pos',int(ybest[te].sum()),'AP',round(average_precision_score(ybest[te],p[te]),4),
              'AUC',round(roc_auc_score(ybest[te],p[te]),4),flush=True)
    d['p_eo']=p
    # Rank within each film sheet removes fold-to-fold calibration shifts.
    d['base_rank']=d.groupby(['region','scene']).score.rank(pct=True)
    d['eo_rank']=d.groupby(['region','scene']).p_eo.rank(pct=True)
    fold_of=d.groupby('region').fold.first().astype(int).to_dict();folds=sorted(set(fold_of.values()))
    gtf={f:to_gt_geojson(G[G.region.map(fold_of)==f]) for f in folds}
    settings=[(w,cap) for w in (0,.25,.5,.75,1.) for cap in (3,5,10,20,50,100,200)]
    counts={}
    for w,cap in settings:
        q=d.assign(z=(1-w)*d.base_rank+w*d.eo_rank).sort_values('z',ascending=False).groupby(['region','scene'],group_keys=False).head(cap).copy()
        q['cls']=KP
        for f in folds:
            _,r=score(fc(q[q.region.map(fold_of)==f]),gtf[f],classes=[KP]);v=r[KP]
            counts[((w,cap),f)]=np.array((v['tp'],v['fp'],v['fn']),int)
    total=np.zeros(3,int);chosen=[]
    for f in folds:
        s=max(settings,key=lambda z:f1(sum((counts[(z,x)] for x in folds if x!=f),np.zeros(3,int))))
        chosen.append(s);total+=counts[(s,f)]
    print('nested',f1(total),tuple(total),chosen)
    for s in sorted(settings,key=lambda z:f1(sum((counts[(z,x)] for x in folds),np.zeros(3,int))),reverse=True)[:15]:
        v=sum((counts[(s,x)] for x in folds),np.zeros(3,int));print('pooled',s,f1(v),tuple(v))
    d[['region','scene','fold','cls','score','x','y','p_eo','y_any','y_best']].to_parquet(CW/'ae_swin_eo_oof.parquet')
if __name__=='__main__':main()
