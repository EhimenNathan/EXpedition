"""Prepare conservative LiDAR-to-aerial pseudo boxes for one Ae geographic fold.

Writes only under Codex Work. No held-out Ae site or its LiDAR labels are used
in training, and a true geodesic 25 km buffer excludes neighbouring sources.
Pseudo labels are exploratory until a site-disjoint model test verifies them.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from pyproj import Geod, Transformer
from scipy.spatial import cKDTree
from shapely.ops import transform as geom_transform

ROOT=Path('/home/nathan/Expedition Competition')
OUT=ROOT/'Codex Work/ae_yolo_pilot/li_ae_pseudo'
CLASSES={'kurgany_povrezhdennye':0,'kurgany_tselye':1}
GEOD=Geod(ellps='WGS84')


def centre(row):
    from rasterio.transform import Affine
    x,y=Affine(*row['transform'])*(row['w']/2,row['h']/2)
    return Transformer.from_crs(row['crs'],4326,always_xy=True).transform(x,y)


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(4));a=p.parse_args()
    ann=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    held=[r for r in ann if r['fold']==a.fold]
    train=[dict(r,boxes=list(r['boxes']),classes=list(r['classes'])) for r in ann if r['fold']!=a.fold]
    sites={r['region'] for r in train}
    held_sites={r['region'] for r in held}
    assert sites.isdisjoint(held_sites)
    hold_lon=np.array([centre(r)[0] for r in held]);hold_lat=np.array([centre(r)[1] for r in held])
    labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    li=labels[(labels['mod']=='Li')&labels.region.isin(sites)&labels.class_name.isin(CLASSES)]
    ae=labels[(labels['mod']=='Ae')&labels.region.isin(sites)&labels.class_name.isin(CLASSES)]
    bysite=defaultdict(list)
    for i,r in enumerate(train):bysite[r['region']].append(i)
    ae_trees={}
    for site,g in ae.groupby('region'):
        xy=np.array([(z.representative_point().x,z.representative_point().y) for z in g.geometry])
        ae_trees[site]=cKDTree(xy) if len(xy) else None
    pix_transform={};lonlat=Transformer.from_crs(3857,4326,always_xy=True)
    candidates=[];audit=defaultdict(int)
    for lab in li.itertuples():
        audit['li_considered']+=1
        site=lab.region;c=lab.geometry.representative_point()
        lon,lat=lonlat.transform(c.x,c.y)
        _,_,dist=GEOD.inv(np.full(len(hold_lon),lon),np.full(len(hold_lon),lat),hold_lon,hold_lat)
        if len(dist) and float(np.min(dist))<25_000:
            audit['within_25km_holdout']+=1;continue
        tree=ae_trees.get(site)
        latitude=2*np.arctan(np.exp(c.y/6378137.))-np.pi/2
        if tree is not None and tree.query([c.x,c.y])[0]*np.cos(latitude)<=10:
            audit['near_existing_ae_label']+=1;continue
        best=None
        for i in bysite[site]:
            r=train[i]
            tf=pix_transform.setdefault(r['crs'],Transformer.from_crs(3857,r['crs'],always_xy=True))
            ux,uy=tf.transform(c.x,c.y)
            a0,b,x0,d,e,y0=r['transform'];px=(ux-x0)/a0;py=(uy-y0)/e
            margin=min(px,py,r['w']-px,r['h']-py)
            if margin<40 or r['valid_frac']<.65:continue
            if best is None or margin>best[0]:best=(margin,i,px,py,tf)
        if best is None:
            audit['not_visible_in_chip']+=1;continue
        _,i,px,py,tf=best;r=train[i]
        im=cv2.imread(str(ROOT/'data/ae_prep/chips'/r['file']),cv2.IMREAD_GRAYSCALE)
        if im is None:continue
        a0,b,x0,d,e,y0=r['transform'];rad=max(20,min(100,round(25/abs(a0))))
        patch=cv2.getRectSubPix(im,(2*rad+1,2*rad+1),(float(px),float(py)))
        valid=patch>2
        spread=float(np.percentile(patch[valid],98)-np.percentile(patch[valid],2)) if valid.sum()>100 else 0.
        if valid.mean()<.8 or spread<35:
            audit['low_visibility']+=1;continue
        bounds=geom_transform(tf.transform,lab.geometry).bounds
        xlo=(bounds[0]-x0)/a0;xhi=(bounds[2]-x0)/a0
        ylo=(bounds[3]-y0)/e;yhi=(bounds[1]-y0)/e
        box=[max(0.,min(xlo,xhi)),max(0.,min(ylo,yhi)),min(float(r['w']),max(xlo,xhi)),min(float(r['h']),max(ylo,yhi))]
        if box[2]-box[0]<3 or box[3]-box[1]<3 or box[2]-box[0]>150 or box[3]-box[1]>150:
            audit['invalid_size']+=1;continue
        candidates.append((site,spread,i,box,CLASSES[lab.class_name]))
    # Limit the influence of dense archaeological complexes. Keep the clearest
    # 40 unmatched LiDAR objects per site; site-uniform training remains intact.
    selected=[]
    for site in sorted({q[0] for q in candidates}):
        selected+=sorted((q for q in candidates if q[0]==site),key=lambda q:q[1],reverse=True)[:40]
    for site,spread,i,box,cls in selected:
        train[i]['boxes'].append([float(v) for v in box])
        train[i]['classes'].append(int(cls))
        audit['added']+=1
    assert all(r['fold']!=a.fold for r in train)
    assert all(r['region'] not in held_sites for r in train)
    OUT.mkdir(exist_ok=True)
    (OUT/f'ann_train_fold{a.fold}.json').write_text(json.dumps(train,ensure_ascii=False))
    summary={'fold':a.fold,'train_sites':len(sites),'heldout_sites':len(held_sites),'audit':dict(audit),
             'site_counts':{s:sum(q[0]==s for q in selected) for s in sorted({q[0] for q in selected})}}
    (OUT/f'audit_fold{a.fold}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
