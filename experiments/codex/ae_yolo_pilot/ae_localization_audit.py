"""CPU read-only audit: damaged Ae misses due to localization vs absent proposals."""
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

ROOT=Path('/home/nathan/Expedition Competition')
D=pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet')
D=D[D.fold.isin((1,2,3))]
L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
L=L[(L['mod']=='Ae')&(L.class_name=='kurgany_povrezhdennye')&(L.region.isin(D.region.unique()))]
for threshold in (.2,.4):
    same=D[(D.cls=='kurgany_povrezhdennye')&(D.score>=threshold)]
    counts=np.zeros(5,int)
    small=np.zeros(5,int)
    for region,gt in L.groupby('region'):
        pred=same[same.region==region]
        if len(pred):
            xy=pred[['x','y']].to_numpy();tree=cKDTree(xy)
        else:tree=None
        for g in gt.geometry:
            c=g.centroid
            lat=np.degrees(2*np.arctan(np.exp(c.y/6378137.))-np.pi/2)
            scale=1/np.cos(np.radians(lat))
            area=g.area/scale**2
            dest=small if area<=100 else counts
            dest[0]+=1
            if tree is None:continue
            # Large polygons can have a centroid far from a boundary. Query a
            # radius enclosing their bounds, then test geometry distances.
            radius=max(c.distance(Point(*g.bounds[:2])),c.distance(Point(*g.bounds[2:])))+11*scale
            cand=tree.query_ball_point((c.x,c.y),radius)
            if not cand:continue
            points=[Point(*xy[j]) for j in cand]
            inside=any(g.contains(p) for p in points)
            d=min(g.distance(p)/scale for p in points)
            dest[1]+=inside
            dest[2]+=d<=2
            dest[3]+=d<=5
            dest[4]+=d<=10
    print('threshold',threshold,'>100m2 GT/inside/<=2m/<=5m/<=10m',counts.tolist(),
          '<=100m2',small.tolist())
