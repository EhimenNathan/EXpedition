"""CPU-only, site-held-out aerial-film DoG proposal and LightGBM pilot.

Train on Ae folds 1 and 2; evaluate once on fold 3. Fold 0 remains untouched.
No pretrained weights, no GPU and no validation-derived threshold selection.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from scipy.spatial import cKDTree

ROOT=Path('/home/nathan/Expedition Competition')
HERE=ROOT/'Codex Work/ae_yolo_pilot'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

SIGMAS=(3.,5.,8.,12.,18.)
CLASSES=('kurgany_povrezhdennye','kurgany_tselye')


def proposals(im, topk=180):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    p2,p98=np.percentile(gray[gray>0],[2,98]) if np.any(gray>0) else (0.,255.)
    spread=max(float(p98-p2),15.)
    smooth=[cv2.GaussianBlur(gray,(0,0),s) for s in (2.,4.,8.,16.,32.)]
    response=np.zeros_like(gray); best_scale=np.zeros_like(gray); polarity=np.zeros_like(gray)
    for sigma in SIGMAS:
        a=cv2.GaussianBlur(gray,(0,0),sigma)
        b=cv2.GaussianBlur(gray,(0,0),1.6*sigma)
        dog=(a-b)/spread
        take=np.abs(dog)>response
        response[take]=np.abs(dog[take]);best_scale[take]=sigma;polarity[take]=np.sign(dog[take])
    valid=(gray>0).astype(np.uint8)
    edge=cv2.distanceTransform(valid,cv2.DIST_L2,3)>max(10,min(gray.shape)/80)
    localmax=(response>=cv2.dilate(response,np.ones((13,13),np.uint8)))&edge
    yy,xx=np.nonzero(localmax)
    if len(xx)>topk:
        keep=np.argpartition(response[yy,xx],-topk)[-topk:]
        yy,xx=yy[keep],xx[keep]
    order=np.argsort(response[yy,xx])[::-1]
    yy,xx=yy[order],xx[order]
    if not len(xx):return np.empty((0,15),np.float32),xx,yy
    # All features are local, contrast-normalized or physical-scale descriptors.
    S=np.array([z[yy,xx] for z in smooth]).T
    local_sq=cv2.GaussianBlur(gray*gray,(0,0),8.)[yy,xx]
    local_var=np.sqrt(np.maximum(local_sq-S[:,2]**2,0))/spread
    lap=cv2.Laplacian(gray,cv2.CV_32F,ksize=3)[yy,xx]/spread
    sobx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    soby=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    grad=np.hypot(sobx[yy,xx],soby[yy,xx])/spread
    features=np.column_stack([
        response[yy,xx],best_scale[yy,xx],polarity[yy,xx],
        (gray[yy,xx]-S[:,4])/spread,
        (S[:,0]-S[:,1])/spread,(S[:,1]-S[:,2])/spread,
        (S[:,2]-S[:,3])/spread,(S[:,3]-S[:,4])/spread,
        local_var,lap,grad,
        np.full(len(xx),spread/255),np.full(len(xx),float(np.mean(gray>0))),
        np.full(len(xx),float(np.mean(gray))/255),
        np.full(len(xx),float(np.std(gray))/255),
    ]).astype(np.float32)
    return features,xx,yy


def labels_for(c,xx,yy):
    # Class 2 = background; exclude candidates in unrelated annotated boxes.
    y=np.full(len(xx),2,np.int8)
    ignored=np.zeros(len(xx),bool)
    for b,k in zip(c['boxes'],c['classes']):
        inside=(xx>=b[0])&(xx<=b[2])&(yy>=b[1])&(yy<=b[3])
        if k in (0,1):
            y[inside]=k
        else:
            ignored|=inside
    return y,~ignored


def merge_radius(P, radius=4.):
    keep=[]
    for (_,cls),d in P.groupby(['region','cls']):
        d=d.sort_values('score',ascending=False)
        xy=d[['x','y']].to_numpy()
        tree=cKDTree(xy)
        used=np.zeros(len(d),bool)
        for i in range(len(d)):
            if used[i]:continue
            keep.append(d.index[i])
            used[tree.query_ball_point(xy[i],radius)]=True
    return P.loc[keep].reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-train',type=int,default=0)
    ap.add_argument('--max-val',type=int,default=0)
    ap.add_argument('--topk',type=int,default=180)
    ap.add_argument('--trees',type=int,default=160)
    a=ap.parse_args()
    C=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    tr=[c for c in C if c['fold'] in (1,2)]
    va=[c for c in C if c['fold']==3]
    if a.max_train:tr=tr[:a.max_train]
    if a.max_val:va=va[:a.max_val]
    Xs=[];Ys=[];rows=[];start=time.time()
    for phase,items in [('train',tr),('val',va)]:
        for i,c in enumerate(items):
            im=cv2.imread(str(ROOT/'data/ae_prep/chips'/c['file']))
            if im is None:raise FileNotFoundError(c['file'])
            x,xx,yy=proposals(im,a.topk)
            if phase=='train':
                y,allow=labels_for(c,xx,yy)
                Xs.append(x[allow]);Ys.append(y[allow])
            else:
                rows.append((c,x,xx,yy))
            if (i+1)%100==0:
                print(phase,i+1,'/',len(items),'elapsed_s',round(time.time()-start,1),flush=True)
    X=np.concatenate(Xs);Y=np.concatenate(Ys)
    print('training candidates',len(X),'labels',np.bincount(Y,minlength=3).tolist(),flush=True)
    model=lgb.LGBMClassifier(n_estimators=a.trees,learning_rate=.04,num_leaves=15,
        max_depth=5,min_child_samples=80,reg_lambda=5.,verbosity=-1,n_jobs=2,
        class_weight={int(k): (8. if k in (0,1) else 1.) for k in np.unique(Y)})
    model.fit(X,Y)
    assert set(model.classes_)=={0,1,2}, 'Need all three classes in training sites'
    pred=[]
    for c,x,xx,yy in rows:
        if not len(x):continue
        p=model.predict_proba(x)
        T=Affine(*c['transform']);trans=Transformer.from_crs(c['crs'],3857,always_xy=True)
        for k in (0,1):
            sel=np.flatnonzero(p[:,k]>=.5)
            for j in sel:
                gx,gy=T*(float(xx[j]),float(yy[j]));mx,my=trans.transform(gx,gy)
                pred.append((c['region'],CLASSES[k],float(p[j,k]),mx,my))
    P=pd.DataFrame(pred,columns=['region','cls','score','x','y'])
    P=merge_radius(P) if len(P) else P
    sites={c['region'] for c in va}
    G=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    G=to_gt_geojson(G[(G['mod']=='Ae')&G.region.isin(sites)])
    overall,res=score(fc(P),G)
    print('fold3 sites',len(sites),'pred',len(P),'official_default_Ae_F1',round(overall,6),flush=True)
    for k in CLASSES:
        print(k,{z:round(v,6) if isinstance(v,float) else v for z,v in res[k].items()},flush=True)
    out=HERE/'blob_pilot';out.mkdir(exist_ok=True)
    P.to_parquet(out/'fold3_predictions.parquet')
    with open(out/'settings.json','w') as f:json.dump(vars(a),f,indent=2)


if __name__=='__main__':main()
