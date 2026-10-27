import warnings; warnings.filterwarnings('ignore')
import json,glob,re,collections,numpy as np, pandas as pd
from shapely.geometry import shape
from shapely.ops import transform as stf
from pyproj import Transformer
CLS={'курганы_поврежденные':'damaged','курганы_целые':'intact','городища':'hillfort','архитектуры':'arch','фортификации':'fort','ObjectPoly':'objects','FindsPoints':'finds'}
rows=[]
trs={}
def tr_for(crs):
    if crs not in trs: trs[crs]=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
    return trs[crs]
for f in glob.glob('*/*/*.geojson'):
    n=f.split('/')[-1][:-8]
    m=re.search(r'_(SpOr|Ae|Ае|Or|Li|La|MagGr|EtGr|TerFl)_',n)
    if not m: continue
    mod={'Ае':'Ae','La':'Li'}.get(m.group(1),m.group(1))
    c=[v for k,v in CLS.items() if n.endswith(k)]
    if not c: continue
    d=json.load(open(f)); nm=(d.get('crs') or {}).get('properties',{}).get('name','EPSG:3857')
    crs='EPSG:4326' if 'CRS84' in nm else 'EPSG:'+nm.split(':')[-1]
    for ft in d['features']:
        if not ft['geometry']: continue
        g=shape(ft['geometry'])
        if g.is_empty: continue
        lon,lat=tr_for(crs).transform(g.representative_point().x,g.representative_point().y)
        # metric area via local UTM
        zone=int((lon+180)//6)+1; utm=f'EPSG:{32600+zone if lat>=0 else 32700+zone}'
        t=Transformer.from_crs(crs,utm,always_xy=True)
        gu=stf(t.transform,g)
        rows.append(dict(region=f.split('/')[0],mod=mod,cls=c[0],gtype=g.geom_type,area=gu.area,lon=lon,lat=lat,
                         x=gu.representative_point().x,y=gu.representative_point().y,utm=utm))
L=pd.DataFrame(rows); L.to_parquet('work/insights/labels_all.parquet')
print(len(L),'labels')
bins=[0,100,500,1000,5000,10000,1e12]; L['bin']=pd.cut(L.area,bins,labels=['<=100','<=500','<=1k','<=5k','<=10k','>10k'])
P=L[L.gtype!='Point']
print('\nPolygon count by class x size bin'); print(pd.crosstab([P.cls],P.bin))
print('\nmedian area (m2) by class x modality'); print(P.pivot_table(index='cls',columns='mod',values='area',aggfunc='median').round(0))
print('\n5-10k bin by class x modality'); print(pd.crosstab(P[P.bin=='<=10k'].cls,P[P.bin=='<=10k'].mod))
print('\n>10k bin by class x modality'); print(pd.crosstab(P[P.bin=='>10k'].cls,P[P.bin=='>10k'].mod))
