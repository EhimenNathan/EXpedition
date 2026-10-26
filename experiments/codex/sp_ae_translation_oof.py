"""Unsupervised per-scene translation test for repeated satellite/Ae observations.

Historical aerial sheets and satellite images have independent georeferencing.
Estimate a translation from OOF proposal point patterns only, then test whether
copying satellite detections into each aerial scene improves exact geographic
OOF.  No ground-truth coordinate is used to fit a transform.
"""
from pathlib import Path
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path('/home/nathan/Expedition Competition');sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'Codex Work')]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc
K=('kurgany_povrezhdennye','kurgany_tselye')

def f1(v):
    tp,fp,fn=v;return 2*tp/max(2*tp+fn+.5*fp,1e-12)

def estimate(S,A,scale,max_m,bin_m):
    if len(S)<2 or len(A)<2:return None
    lim=max_m*scale; ds=[]
    tree=cKDTree(A)
    for s in S:
        for j in tree.query_ball_point(s,lim): ds.append(A[j]-s)
    if len(ds)<2:return None
    ds=np.asarray(ds); b=np.round(ds/(bin_m*scale)).astype(int)
    u,n=np.unique(b,axis=0,return_counts=True); seed=u[n.argmax()]*bin_m*scale
    q=ds[np.linalg.norm(ds-seed,axis=1)<=1.5*bin_m*scale]
    if len(q)<2:return None
    delta=np.median(q,axis=0)
    # One-to-one-ish support after the robust translation.
    ta=cKDTree(A); dist,idx=ta.query(S+delta)
    good=np.where(dist<=12*scale)[0]
    if len(good)<2:return delta,len(good),float(np.median(dist[good])) if len(good) else 1e9
    # refine with unique target matches, strongest residual first
    order=good[np.argsort(dist[good])]; seen=set(); keep=[]
    for i in order:
        if int(idx[i]) not in seen:seen.add(int(idx[i]));keep.append(i)
    delta=np.median(A[idx[keep]]-S[keep],axis=0)
    return delta,len(keep),float(np.median(np.linalg.norm(A[idx[keep]]-(S[keep]+delta),axis=1)))

def main():
    sat=pd.read_parquet(ROOT/'Codex Work/sp_region_pooled_five_oof.parquet')
    ae=pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_layers_cands.parquet')
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    common=sorted(set(sat.region)&set(ae.region)&set(L[L['mod']=='Ae'].region))
    sat=sat[sat.region.isin(common)&sat.cls.isin(K)&(sat.score>=.2)][['region','cls','score','x','y']].copy()
    ae=ae[ae.region.isin(common)&ae.cls.isin(K)].copy(); fold=ae.groupby('region').fold.first().astype(int).to_dict();folds=sorted(set(fold.values()))
    sg=L[(L['mod']=='SpOr')&L.region.isin(common)&L.class_name.isin(K)]
    ag=layer_gt(L[(L['mod']=='Ae')&L.region.isin(common)&L.class_name.isin(K)])
    G=gpd.GeoDataFrame(pd.concat([sg,ag],ignore_index=True),crs=L.crs);GT={f:to_gt_geojson(G[G.region.map(fold)==f]) for f in folds}
    settings=[(a,m,b,k,mode) for a in (.08,.12,.16,.20) for m in (50,100,200) for b in (5,10,20) for k in (2,3,5) for mode in ('confirmed','all')]
    cache={}
    def pred(st):
        athr,maxm,binm,need,mode=st;add=[]
        for region in common:
            s=sat[sat.region==region];S=s[['x','y']].to_numpy(float)
            if not len(S):continue
            scale=1/np.cos(np.radians(np.degrees(2*np.arctan(np.exp(np.median(S[:,1])/6378137))-np.pi/2)))
            for scene,q0 in ae[(ae.region==region)&(ae.score>=athr)].groupby('scene'):
                A=q0[['x','y']].to_numpy(float); z=estimate(S,A,scale,maxm,binm)
                if z is None or z[1]<need:continue
                d,sup,res=z; tree=cKDTree(A)
                for row,xy in zip(s.itertuples(index=False),S+d):
                    dist,j=tree.query(xy)
                    if mode=='confirmed' and dist>12*scale:continue
                    out=A[j] if dist<=12*scale else xy
                    add.append((region,row.cls,row.score,out[0],out[1],scene,sup,res,d[0]/scale,d[1]/scale))
        cols=['region','cls','score','x','y','scene','support','resid','dx_m','dy_m']
        d=pd.DataFrame(add,columns=cols)
        return pd.concat([sat,d[sat.columns]],ignore_index=True),d
    counts={}
    for i,st in enumerate(settings):
        p,_=pred(st)
        for f in folds:
            _,rr=score(fc(p[p.region.map(fold)==f]),GT[f],classes=list(K))
            counts[(st,f)]={c:np.array([rr[c]['tp'],rr[c]['fp'],rr[c]['fn']],int) for c in K}
        if i%75==0:print('setting',i,st,flush=True)
    util=lambda d:(30*f1(d[K[0]])+20*f1(d[K[1]]))/50
    total={c:np.zeros(3,int) for c in K};chosen={}
    for f in folds:
        st=max(settings,key=lambda x:util({c:sum((counts[(x,g)][c] for g in folds if g!=f),np.zeros(3,int)) for c in K}))
        chosen[f]=st
        for c in K:total[c]+=counts[(st,f)][c]
    ranked=[]
    for st in settings:
        d={c:sum((counts[(st,f)][c] for f in folds),np.zeros(3,int)) for c in K};ranked.append((util(d),st,d))
    ranked.sort(reverse=True,key=lambda x:x[0]);print('nested',util(total),'chosen',chosen)
    for c in K:print(c,f1(total[c]),tuple(total[c]))
    for u,st,d in ranked[:12]:print('pooled',u,st,{c:(f1(d[c]),tuple(d[c])) for c in K})
    best=ranked[0][1];_,D=pred(best);D.to_parquet(ROOT/'Codex Work/sp_ae_translation_best.parquet',index=False)
    print('best transforms/additions');print(D.groupby(['region','scene'])[['support','resid','dx_m','dy_m']].first().to_string() if len(D) else 'none')

if __name__=='__main__':main()
