"""CPU-only aerial-film morphology reranker on site-disjoint OOF candidates.

The classifier sees image pixels, detector score and class, but no site ID or
coordinates. Folds 1-3 select a fixed score policy; fold 0 is the final audit.
"""
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point

ROOT = Path('/home/nathan/Expedition Competition')
HERE = ROOT / 'Codex Work/ae_yolo_pilot'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc

CLASSES = ('kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha',
           'arkhitektury', 'fortifikatsii')
RADII = np.array([0., 5., 10., 18., 30., 45.], dtype=np.float32)
OUT = HERE / 'radial_pilot'


@lru_cache(maxsize=8)
def image(file):
    im = cv2.imread(str(ROOT / 'data/ae_prep/chips' / file), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(file)
    return im


def crop_features(im, px, py, gsd):
    # 45 metres in the source scene, resampled to a common physical grid.
    radius = max(8, min(200, int(round(45. / gsd))))
    h, w = im.shape
    x, y = int(round(px)), int(round(py))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    patch = cv2.getRectSubPix(im, (2*radius+1, 2*radius+1), (float(x), float(y)))
    z = cv2.resize(patch, (91,91), interpolation=cv2.INTER_AREA).astype(np.float32)
    valid = z > 2
    if valid.mean() < .7:
        return None
    lo, hi = np.percentile(z[valid], [2,98])
    z = np.clip((z-lo)/max(hi-lo,1.),0,1).astype(np.float32)
    yy, xx = np.mgrid[-45:46,-45:46]
    rr = np.hypot(xx,yy)
    feats=[]
    for a,b in zip(RADII[:-1],RADII[1:]):
        q=z[(rr>=a)&(rr<b)]
        feats += [float(np.mean(q)), float(np.std(q)), float(np.percentile(q,10)),
                  float(np.percentile(q,90))]
    lap=cv2.Laplacian(z,cv2.CV_32F)
    gx=cv2.Sobel(z,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(z,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(gx,gy)
    for a,b in ((0,10),(10,30),(30,45)):
        mask=(rr>=a)&(rr<b)
        feats += [float(np.mean(np.abs(lap[mask]))),float(np.mean(mag[mask]))]
    # Centre-surround and ring extrema are rotation invariant.
    mu=[feats[4*i] for i in range(5)]
    feats += [mu[0]-mu[2],mu[1]-mu[3],mu[2]-mu[4],float(hi-lo),float(gsd)]
    return feats


def make_features():
    raw=pd.read_parquet(HERE/'fusion_pilot/equal_candidates.parquet')
    raw=raw[raw.score>=.12].sort_values('score',ascending=False).groupby(
        ['region','cls'],sort=False).head(150).reset_index(drop=True)
    ann=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    bysite=defaultdict(list)
    for r in ann:
        a,b,c,d,e,f=r['transform']
        bysite[r['region']].append((r,a,c,e,f))
    labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    labels=labels[(labels['mod']=='Ae') & labels.region.isin(raw.region)]
    bygt=defaultdict(list)
    for r in labels.itertuples():
        bygt[(r.region,r.class_name)].append(r.geometry)
    transforms={}
    output=[]
    for region, rows in raw.groupby('region',sort=False):
        chips=bysite[region]
        if not chips:
            continue
        crs=chips[0][0]['crs']
        tr=transforms.setdefault(crs,Transformer.from_crs(3857,crs,always_xy=True))
        X,Y=tr.transform(rows.x.to_numpy(),rows.y.to_numpy())
        for row,ux,uy in zip(rows.itertuples(),X,Y):
            best=None
            for r,a,c,e,f in chips:
                px=(ux-c)/a;py=(uy-f)/e
                if 0<=px<r['w'] and 0<=py<r['h']:
                    margin=min(px,py,r['w']-px,r['h']-py)
                    if best is None or margin>best[0]:
                        best=(margin,r,px,py,abs(a))
            if best is None:
                continue
            _,r,px,py,gsd=best
            feats=crop_features(image(r['file']),px,py,gsd)
            if feats is None:
                continue
            # Label is used only for training-side candidate quality learning.
            # The final fold remains entirely unseen until evaluation.
            k=1/np.cos(np.radians(np.degrees(2*np.arctan(np.exp(row.y/6378137.))-np.pi/2)))
            pt=Point(row.x,row.y)
            y=int(any(g.distance(pt)<=4*k for g in bygt[(region,row.cls)]))
            output.append((row.Index, y, *feats))
        print('features',region,len(output),flush=True)
    cols=['idx','label']+[f'pixel_{i}' for i in range(len(output[0])-2)]
    F=pd.DataFrame(output,columns=cols)
    F=raw.loc[F.idx].reset_index(drop=True).join(F.drop(columns='idx').reset_index(drop=True))
    OUT.mkdir(exist_ok=True)
    F.to_parquet(OUT/'features.parquet',index=False)
    print('FEATURES',len(F),'near_gt',int(F.label.sum()),'folds',F.fold.value_counts().to_dict())
    return F


def evaluate(F):
    import lightgbm as lgb
    labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    site_fold={r['region']:r['fold'] for r in json.loads((ROOT/'data/ae_prep/ann.json').read_text())}
    labels=labels[(labels['mod']=='Ae') & labels.region.isin(site_fold)]
    pred=F.copy()
    pred['quality']=np.nan
    cols=['score']+[c for c in F if c.startswith('pixel_')]
    for c in CLASSES:
        pred[f'cls_{c}']=(pred.cls==c).astype(np.int8)
        cols.append(f'cls_{c}')
    for hold in range(4):
        training_folds=({1,2,3}-{hold}) if hold else {1,2,3}
        tr=pred[pred.fold.isin(training_folds)];va=pred[pred.fold==hold]
        model=lgb.LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=4,
            learning_rate=.035,min_child_samples=80,colsample_bytree=.8,
            reg_lambda=5.,verbose=-1,n_jobs=4,random_state=20261024)
        model.fit(tr[cols],tr.label)
        pred.loc[va.index,'quality']=model.predict_proba(va[cols])[:,1]
    assert pred.quality.notna().all()
    policies={'base':lambda d:d.score,
              'quality':lambda d:d.quality,
              'product':lambda d:d.score*np.sqrt(d.quality),
              'blend':lambda d:.5*d.score+.5*d.quality}
    results={}
    for hold in range(4):
        sites={site for site,fold in site_fold.items() if fold==hold}
        gt=to_gt_geojson(labels[labels.region.isin(sites)])
        d=pred[pred.fold==hold]
        for name,fn in policies.items():
            q=d.copy();q['score']=fn(q)
            for n in (30,60,100):
                sel=q.sort_values('score',ascending=False).groupby(['region','cls'],sort=False).head(n)
                s,res=score(fc(sel),gt)
                results[(hold,name,n)]=(s,res['kurgany_povrezhdennye']['f1'])
                print('fold',hold,name,n,'AeF1',round(s,5),'damagedF1',round(results[(hold,name,n)][1],5),flush=True)
    for hold in range(4):
        train=[f for f in (1,2,3) if f!=hold]
        if hold==0:train=[1,2,3]
        options=[(name,n) for name in policies for n in (30,60,100)]
        choice=max(options,key=lambda x:np.mean([results[(f,*x)][0] for f in train]))
        print('heldout',hold,'selection',choice,'score',results[(hold,*choice)],flush=True)


if __name__=='__main__':
    p=OUT/'features.parquet'
    F=pd.read_parquet(p) if p.exists() else make_features()
    evaluate(F)
