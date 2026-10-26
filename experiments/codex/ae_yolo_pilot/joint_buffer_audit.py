"""Read-only audit of the 25 km joint Ae/SpOr holdout exclusion.

The training code measures its threshold in EPSG:3857 metres. This script
compares that with ellipsoidal distances without reading validation labels.
"""
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer, Geod

ROOT=Path('/home/nathan/Expedition Competition/data')
AE=json.loads((ROOT/'ae_prep/ann.json').read_text())
SP=[r for r in json.loads((ROOT/'spor_prep/ann.json').read_text()) if r['boxes']]
GEOD=Geod(ellps='WGS84')

def centers(rows):
    a=[]
    for r in rows:
        t=r['transform']; x=t[0]*r['w']/2+t[1]*r['h']/2+t[2]
        y=t[3]*r['w']/2+t[4]*r['h']/2+t[5]
        lon,lat=Transformer.from_crs(r['crs'],4326,always_xy=True).transform(x,y)
        mx,my=Transformer.from_crs(4326,3857,always_xy=True).transform(lon,lat)
        a.append((mx,my,lon,lat))
    return np.asarray(a,float)

A=centers(AE);S=centers(SP)
for fold in range(4):
    held=A[np.array([r['fold']==fold for r in AE])]
    merc=np.linalg.norm(S[:,None,:2]-held[None,:,:2],axis=2)
    mmin=merc.min(axis=1)
    passed=mmin>=25000.
    geo=np.empty(len(S))
    for i,(mx,my,lon,lat) in enumerate(S):
        _,_,distance=GEOD.inv(np.full(len(held),lon),np.full(len(held),lat),held[:,2],held[:,3])
        geo[i]=distance.min()
    bad=np.flatnonzero(passed&(geo<25000.))
    print('fold',fold,'held_ae_chips',len(held),'positive_spor',len(S),
          'passed_projected',int(passed.sum()),'inside_real_25km',len(bad),
          'min_real_distance_km',round(float(geo[passed].min()/1000),3))
    for i in bad[:5]:
        print(' example',SP[i]['region'],'merc_km',round(mmin[i]/1000,2),
              'geodesic_km',round(geo[i]/1000,2))
