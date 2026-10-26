import os, json, glob, numpy as np, geopandas as gpd, pandas as pd, rasterio, warnings; warnings.filterwarnings("ignore")
from rasterio.windows import from_bounds
from PIL import Image, ImageDraw
os.environ.update(AWS_S3_ENDPOINT="s3.ru-7.storage.selcloud.ru",
  AWS_ACCESS_KEY_ID=os.environ["KOZ3_S3_ACCESS_KEY"],
  AWS_SECRET_ACCESS_KEY=os.environ["KOZ3_S3_SECRET_KEY"], AWS_VIRTUAL_HOSTING="FALSE", AWS_REGION="ru-7", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
T=pd.read_csv("docs/tif_headers.csv")
COL={"курганы_целые":(255,196,61),"курганы_поврежденные":(255,120,70),"городища":(90,200,255),"фортификации":(190,120,255),"архитектуры":(120,255,170)}
def key(site,pat): return [k for k in T.key if k.startswith(site+"/") and pat in k][0]
def utm(site): return json.load(open(f"data_meta/{site}/UTM.json",encoding="utf-8"))["crs"].replace("urn:ogc:def:crs:EPSG::","EPSG:")
def labels(site,mod):
    L=[]
    for f in glob.glob(f"data_meta/{site}/**/*.geojson",recursive=True):
        b=os.path.basename(f).replace("Ае","Ae")
        if f"_{mod}_" not in b: continue
        g=gpd.read_file(f); g["cls"]=b.split(f"_{mod}_")[-1][:-8]; L.append(g)
    return pd.concat(L) if L else None
def render(name,site,pat,mod,cx,cy,half,out_px=900,outline=True,aspect=1.0,stretch=True):
    k=key(site,pat); crs_site=utm(site)
    with rasterio.open(f"/vsis3/train-expds-3/{k}") as r:
        rc=r.crs if (r.crs is not None and r.crs.is_epsg_code) else crs_site
        pts=gpd.GeoSeries(gpd.points_from_xy([cx-half,cx+half],[cy-half*aspect,cy+half*aspect]),crs=crs_site).to_crs(rc)
        x0,x1=pts.x.min(),pts.x.max(); y0,y1=pts.y.min(),pts.y.max()
        w=from_bounds(x0,y0,x1,y1,r.transform)
        H=int(out_px*aspect); W=out_px
        a=r.read(indexes=list(range(1,min(3,r.count)+1)),window=w,out_shape=(min(3,r.count),H,W),boundless=True,fill_value=0,resampling=rasterio.enums.Resampling.average).astype(np.float32)
    if stretch:
        for i in range(a.shape[0]):
            v=a[i][a[i]>0]; lo,hi=(np.percentile(v,[1,99.5]) if v.size else (0,1)); a[i]=np.clip((a[i]-lo)/(hi-lo+1e-6)*255,0,255)
    img=np.moveaxis(a,0,-1).astype(np.uint8)
    if img.shape[2]==1: img=np.repeat(img,3,2)
    im=Image.fromarray(img); d=ImageDraw.Draw(im)
    L=labels(site,mod) if outline else None
    if L is not None:
        L=L.to_crs(rc)
        def px(x,y): return ((x-x0)/(x1-x0)*W,(y1-y)/(y1-y0)*H)
        for geom,c in zip(L.geometry,L.cls):
            if geom is None or geom.is_empty: continue
            for p in getattr(geom,"geoms",[geom]):
                if p.geom_type!="Polygon": continue
                xy=[px(x,y) for x,y in p.exterior.coords]
                if all(0<=u<=W and 0<=v<=H for u,v in xy[:1]) or True: d.line(xy,fill=COL.get(c,(255,255,255)),width=max(2,W//300))
    im.save(f"deck_img/{name}.png"); print(name,im.size)
jobs=[("li_timerevo","036_Тимерево","_Li_g","Li",544204.2,6377426.5,45),
      ("ae_timerevo","036_Тимерево","_Ae_","Ae",544204.2,6377426.5,45),
      ("lic_timerevo","036_Тимерево","_Li_c","Li",544204.2,6377426.5,45),
      ("li_fort_borovichi","004_Боровичи_1","_Li_g","Li",548298.7,6473408.7,45),
      ("li_hillfort_demidovka","015_Демидовка","_Li_g","Li",426328.7,6067141.3,90),
      ("sp_kurgans_kalmykia","048_Калмыкия_1","01_Калмыкия_1_SpOr","SpOr",447409.7,5135165.9,160),
      ("sp_kurgans_kbr","057_Кабардино_Балкария","01_Кабардино_Балкария_SpOr","SpOr",400124.6,4818270.9,160),
      ("sp_arch_kislovodsk","164_Кисловодская_котловина_1","_SpOr_","SpOr",290043.9,4856235.9,80),
      ("sp_hillfort_eliz","033_Елизаветинское","_SpOr_","SpOr",484798.5,4986965.7,170),
      ("or_hillfort_eliz","033_Елизаветинское","_Or_","Or",484861.2,4986923.1,120)]
import sys
sel=sys.argv[1:] or [j[0] for j in jobs]
for j in jobs:
    if j[0] in sel:
        try: render(*j)
        except Exception as e: print("FAIL",j[0],e)

def autocenter(site,pat,mod):
    k=key(site,pat); crs_site=utm(site)
    with rasterio.open(f"/vsis3/train-expds-3/{k}") as r:
        rc=r.crs if (r.crs is not None and r.crs.is_epsg_code) else crs_site; b=r.bounds
    L=labels(site,mod).to_crs(rc); c=L.geometry.centroid
    inside=c[(c.x>b.left)&(c.x<b.right)&(c.y>b.bottom)&(c.y<b.top)]
    import numpy as np
    xy=np.c_[inside.x,inside.y]; d=((xy[:,None]-xy[None])**2).sum(-1)**.5
    best=xy[(d<150).sum(1).argmax()]
    p=gpd.GeoSeries(gpd.points_from_xy([best[0]],[best[1]]),crs=rc).to_crs(crs_site)
    return float(p.x[0]),float(p.y[0])
