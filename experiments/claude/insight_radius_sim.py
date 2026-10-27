import json,glob,re,warnings,numpy as np,pandas as pd; warnings.filterwarnings('ignore')
from shapely.geometry import shape, Point
from shapely.ops import transform as stf, polylabel
from pyproj import Transformer
from multiprocessing import Pool
D='data'
CLS={'курганы_поврежденные':'damaged','курганы_целые':'intact','городища':'hillfort','архитектуры':'arch','фортификации':'fort','ObjectPoly':'objects'}
def crs_of(d):
    nm=(d.get('crs') or {}).get('properties',{}).get('name','EPSG:3857'); return 'EPSG:4326' if 'CRS84' in nm else 'EPSG:'+nm.split(':')[-1]
def load(f):
    n=f.split('/')[-1][:-8]; m=re.search(r'_(SpOr|Ae|Ае|Or|Li|MagGr|EtGr|TerFl)_',n)
    c=[v for k,v in CLS.items() if n.endswith(k)]
    if not m or not c: return []
    d=json.load(open(f)); out=[]
    for ft in d['features']:
        if not ft['geometry']: continue
        g=shape(ft['geometry'])
        if g.is_empty or g.geom_type not in ('Polygon','MultiPolygon'): continue
        p=g.representative_point(); lon,lat=Transformer.from_crs(crs_of(d),'EPSG:4326',always_xy=True).transform(p.x,p.y)
        utm=f'EPSG:{32600+int((lon+180)//6)+1}'
        g=stf(Transformer.from_crs(crs_of(d),utm,always_xy=True).transform,g)
        if g.geom_type=='MultiPolygon': g=max(g.geoms,key=lambda q:q.area)
        if not g.is_valid: g=g.buffer(0)
        out.append(({'Ае':'Ae'}.get(m.group(1),m.group(1)),c[0],g))
    return out
RADII=(0.25,0.5,1.0,1.5,2.0); SIG=(0.0,0.5,1.0,2.0)
def hits(item):
    mod,cls,g=item; rng=np.random.default_rng(hash(g.wkb)%2**32)
    try: a=polylabel(g,0.05)                        # best interior anchor (pole of inaccessibility)
    except Exception: a=g.representative_point()
    inr=g.boundary.distance(a)
    res={}
    for s in SIG:
        pts=[(a.x+e[0],a.y+e[1]) for e in rng.normal(0,s,(20 if s else 1,2))]
        for r in RADII:
            h=0
            for x,y in pts:
                c=Point(x,y).buffer(r,32); inter=c.intersection(g).area
                h+= (inter/min(c.area,g.area)>=0.5) and g.contains(Point(x,y))
            res[(s,r)]=h/len(pts)
    return mod,cls,g.area,inr,res
if __name__=='__main__':
    items=[]
    for f in glob.glob(f'{D}/*/*/*.geojson'): items+=load(f)
    with Pool(4) as p: R=p.map(hits,items,chunksize=50)
    rows=[dict(mod=m,cls=c,area=a,inr=i,**{f's{s}_r{r}':v for (s,r),v in res.items()}) for m,c,a,i,res in R]
    T=pd.DataFrame(rows); T.to_pickle('work/insights/radius_sim.pkl')
    print(len(T),'polygon labels')
    print('\ninscribed radius (m): share < 1 m by class x type'); print(T.assign(thin=T.inr<1).pivot_table(index='cls',columns='mod',values='thin',aggfunc='mean').round(2))
    for s in SIG:
        print(f'\nhit rate with placement error σ={s} m, by circle radius r (rows: class)')
        print(T.groupby('cls')[[f's{s}_r{r}' for r in RADII]].mean().round(3).rename(columns=lambda c:'r='+c.split('_r')[1]))
