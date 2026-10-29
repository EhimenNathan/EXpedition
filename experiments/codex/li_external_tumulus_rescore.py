#!/usr/bin/env python3
"""Score deployed LiDAR kurgan peaks with the released DK/RO mound CNN.

This isolates whether the external 21k-mound training corpus adds ranking signal
at proposals already found by our much stronger detector. No parameters are fit
here; the primary verdict is site-held-out AUC and incremental OOF AUC.
"""
from pathlib import Path
import os, sys, gc
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXT=Path(os.environ.get('TUMULUS_EXT','/home/nathan/Expedition Competition/work/external/tumulus-lidar-detector'))
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/claude'),str(EXT/'tools'),str(ROOT/'scripts')]

def main():
    from koz3.li_train import labels,site_items
    from li_dome_features import load_dtm
    from pyproj import Transformer
    import tumul_scan as TS
    import lib_channels as LC
    import torch
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    src=ROOT/'work/li/rescore_cands.parquet'; out=ROOT/'work/li_external_tumulus_rescore.parquet'
    D=pd.read_parquet(src)
    if out.exists():
        E=pd.read_parquet(out); D=D.merge(E[['region','x','y','ext_cnn']],on=['region','x','y'],how='left')
    else:
        items={x['region']:x for x in site_items(labels())}; parts=[]
        for i,(region,G) in enumerate(D.groupby('region'),1):
            it=items[region]; d,v,x0,ytop=load_dtm(region); d=np.where(v,d,np.nan)
            fill=np.where(np.isfinite(d),d,float(np.nanmedian(d)))
            tr=Transformer.from_crs(3857,it['meta']['crs'],always_xy=True)
            xx,yy=tr.transform(G.x.values,G.y.values); scores=np.zeros(len(G),np.float32)
            batch=[]; idx=[]; hw=int(40/.5)
            def flush():
                if not batch:return
                X=torch.tensor(np.asarray(batch,np.uint8)).float()/255.
                with torch.no_grad(): val=torch.sigmoid(TS.net()(X)).cpu().numpy()
                scores[np.asarray(idx)]=val; batch.clear();idx.clear()
            for j,(X,Y) in enumerate(zip(xx,yy)):
                px=int((X-x0)/.5); py=int((ytop-Y)/.5)
                w=fill[py-hw:py+hw,px-hw:px+hw]
                if w.shape!=(2*hw,2*hw):continue
                s=LC.stamp_multi(w,.5,TS.CHANS)
                if s is not None:batch.append(s);idx.append(j)
                if len(batch)>=256:flush()
            flush(); q=G[['region','x','y']].copy();q['ext_cnn']=scores;parts.append(q)
            print(f'[{i}/{D.region.nunique()}] {region}: {len(G)} peaks, ext mean {scores.mean():.3f}',flush=True)
            del d,v,fill;gc.collect()
        E=pd.concat(parts,ignore_index=True);E.to_parquet(out,index=False)
        D=D.merge(E,on=['region','x','y'],how='left')
    D.ext_cnn=D.ext_cnn.fillna(0.)
    print(f"AUC peak {roc_auc_score(D.lab,D.peak):.4f} conf {roc_auc_score(D.lab,D.conf):.4f} external {roc_auc_score(D.lab,D.ext_cnn):.4f}")
    base=['peak','p_dam','p_int','p_ratio','conf']
    for cols,name in ((base,'small base'),(base+['ext_cnn'],'small base + external')):
        D['p']=0.
        for f in sorted(D.fold.unique()):
            tr=D.fold!=f;te=~tr
            m=lgb.LGBMClassifier(n_estimators=200,learning_rate=.03,num_leaves=7,min_child_samples=60,verbosity=-1).fit(D.loc[tr,cols],D.loc[tr,'lab'])
            D.loc[te,'p']=m.predict_proba(D.loc[te,cols])[:,1]
        print(f"OOF AUC {name:24s} {roc_auc_score(D.lab,D.p):.4f} per-fold "+' '.join(f"{f}:{roc_auc_score(D.loc[D.fold==f,'lab'],D.loc[D.fold==f,'p']):.3f}" for f in sorted(D.fold.unique())))

if __name__=='__main__':main()
