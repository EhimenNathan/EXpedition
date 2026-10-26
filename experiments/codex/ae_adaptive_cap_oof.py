"""Low-parameter, scene-adaptive candidate budget for dense aerial cemeteries.

The historical-film detector has radically different precision by source
sheet.  This tests whether observable proposal density can safely allocate a
larger emission budget to cemetery-like sheets.  Every rule is selected on
other geographic folds and evaluated with source-image-isolated GT.
"""
from pathlib import Path
import sys
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition');CW=ROOT/'Codex Work'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(CW)]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc
KP='kurgany_povrezhdennye'

def f1(v):
    tp,fp,fn=v;return 2*tp/max(2*tp+fn+.5*fp,1e-12)

def main():
    d=pd.read_parquet(CW/'ae_sheet_selfadapt_features.parquet')
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=L[(L['mod']=='Ae')&L.region.isin(d.region.unique())]
    G=layer_gt(L); fold=d.groupby('region').fold.first().astype(int).to_dict();folds=sorted(set(fold.values()))
    GT={f:to_gt_geojson(G[G.region.map(fold)==f]) for f in folds}
    by={k:g.sort_values('score',ascending=False).copy() for k,g in d.groupby(['region','scene'])}
    feats=[]
    for (r,s),g in by.items():
        feats.append(dict(region=r,scene=s,fold=fold[r],n=len(g),n20=(g.score>=.2).sum(),n30=(g.score>=.3).sum(),n40=(g.score>=.4).sum(),n50=(g.score>=.5).sum()))
    F=pd.DataFrame(feats)
    settings=[]
    for col in ('n20','n30','n40','n50'):
      for cut in (10,20,40,80):
       for lo in (20,50,100):
        for hi in (200,400):
         if hi>lo:settings.append((col,cut,lo,hi))
    # Include invariant controls.
    settings += [('n',10**9,c,c) for c in (0,20,50,100,200,400,800)]
    counts={}
    for i,st in enumerate(settings):
        col,cut,lo,hi=st;pp=[]
        for z in F.itertuples(index=False):
            cap=hi if getattr(z,col)>=cut else lo
            if cap:
                q=by[(z.region,z.scene)].head(cap).copy();q['cls']=KP;pp.append(q)
        p=pd.concat(pp,ignore_index=True) if pp else d.iloc[:0].copy()
        for f in folds:
            _,rr=score(fc(p[p.region.map(fold)==f]),GT[f],classes=[KP]);v=rr[KP]
            counts[(st,f)]=np.array([v['tp'],v['fp'],v['fn']],int)
        if i%150==0:print('setting',i,st,flush=True)
    total=np.zeros(3,int);chosen={}
    for f in folds:
        st=max(settings,key=lambda s:f1(sum((counts[(s,g)] for g in folds if g!=f),np.zeros(3,int))))
        chosen[f]=st;total+=counts[(st,f)]
    ranked=[]
    for st in settings:
        c=sum((counts[(st,f)] for f in folds),np.zeros(3,int));ranked.append((f1(c),st,c))
    ranked.sort(reverse=True,key=lambda x:x[0])
    print('scenes',len(F),'nested',f1(total),tuple(total),'chosen',chosen)
    for x in ranked[:20]:print('pooled',x[0],x[1],tuple(x[2]))
    print('scene features');print(F.sort_values('n30',ascending=False).to_string(index=False))

if __name__=='__main__':main()
