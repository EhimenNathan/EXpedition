"""Unlabelled image-based center refinement for Ae detector candidates.

Tests one physically fixed 3 m bright-blob snap and one absolute-blob snap
against the original OOF centers. No model training or GPU use.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc


def response(gray):
    gray=gray.astype(np.float32)
    spread=max(float(np.percentile(gray[gray>0],95)-np.percentile(gray[gray>0],5)),15.)
    bright=np.zeros_like(gray);absolute=np.zeros_like(gray)
    for s in (3.,5.,8.,12.):
        dog=(cv2.GaussianBlur(gray,(0,0),s)-cv2.GaussianBlur(gray,(0,0),1.6*s))/spread
        bright=np.maximum(bright,dog)
        absolute=np.maximum(absolute,np.abs(dog))
    return bright,absolute


def main():
    Dfull=pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet')
    Dfull=Dfull[Dfull.fold.isin((1,2,3))]
    F=Dfull[['region','fold']].drop_duplicates().set_index('region').fold.to_dict()
    D=Dfull[Dfull.score>=.4].copy().reset_index(drop=True)
    C=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    by_site=defaultdict(list)
    for c in C:
        if c['fold'] in (1,2,3):
            c['T']=Affine(*c['transform']);c['invT']=~c['T']
            c['to_chip']=Transformer.from_crs(3857,c['crs'],always_xy=True)
            c['to_m']=Transformer.from_crs(c['crs'],3857,always_xy=True)
            c['gsd']=float(abs(c['T'].a*c['T'].e)**.5)
            by_site[c['region']].append(c)
    assignments=defaultdict(list);unassigned=0
    for row in D.itertuples():
        best=None;best_margin=-1
        for c in by_site[row.region]:
            gx,gy=c['to_chip'].transform(row.x,row.y)
            px,py=c['invT']*(gx,gy)
            margin=min(px,py,c['w']-px,c['h']-py)
            if margin>12 and margin>best_margin:
                best=(c,px,py);best_margin=margin
        if best is None:
            unassigned+=1;continue
        c,px,py=best
        assignments[c['file']].append((row.Index,px,py))
    brightxy=D[['x','y']].to_numpy().copy()
    absxy=brightxy.copy()
    by_file={c['file']:c for c in C if c['fold'] in (1,2,3)}
    for n,(file,items) in enumerate(assignments.items(),1):
        c=by_file[file]
        image=cv2.imread(str(ROOT/'data/ae_prep/chips'/file),cv2.IMREAD_GRAYSCALE)
        if image is None:raise FileNotFoundError(file)
        b,a=response(image)
        rad=min(18,max(3,round(3./c['gsd'])))
        for idx,px,py in items:
            x0=max(0,int(round(px))-rad);x1=min(image.shape[1],int(round(px))+rad+1)
            y0=max(0,int(round(py))-rad);y1=min(image.shape[0],int(round(py))+rad+1)
            for layer,arr in ((b,brightxy),(a,absxy)):
                roi=layer[y0:y1,x0:x1]
                yy,xx=np.unravel_index(np.argmax(roi),roi.shape)
                gx,gy=c['T']*(x0+xx,y0+yy)
                arr[idx]=c['to_m'].transform(gx,gy)
        if n%100==0:print('images',n,'/',len(assignments),flush=True)
    print('assigned',len(D)-unassigned,'unassigned',unassigned)
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='Ae')&L.region.isin(F)]
    for name,xy in [('base',D[['x','y']].to_numpy()),('bright',brightxy),('absolute',absxy)]:
        Q=D.copy();Q[['x','y']]=xy
        for fold in (1,2,3):
            g=to_gt_geojson(L[L.region.map(F)==fold])
            s,r=score(fc(Q[Q.fold==fold]),g)
            d=r['kurgany_povrezhdennye']
            print(name,'fold',fold,'AeF1',round(s,6),'damaged',
                  (d['tp'],d['fp'],d['fn']),flush=True)
        out=ROOT/'Codex Work/ae_yolo_pilot/snap_pilot';out.mkdir(exist_ok=True)
        Q.to_parquet(out/f'{name}.parquet')


if __name__=='__main__':main()
