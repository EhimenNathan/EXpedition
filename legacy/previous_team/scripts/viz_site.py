import os, sys, rasterio, numpy as np, geopandas as gpd, glob, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from rasterio.crs import CRS
os.environ.update(AWS_S3_ENDPOINT="s3.ru-7.storage.selcloud.ru",
  AWS_ACCESS_KEY_ID=os.environ["KOZ3_S3_ACCESS_KEY"],
  AWS_SECRET_ACCESS_KEY=os.environ["KOZ3_S3_SECRET_KEY"], AWS_VIRTUAL_HOSTING="FALSE", AWS_REGION="ru-7", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
site=sys.argv[1]; half=float(sys.argv[2]) if len(sys.argv)>2 else 60
utm=json.load(open(f"data_meta/{site}/UTM.json",encoding="utf-8"))["crs"].replace("urn:ogc:def:crs:EPSG::","EPSG:")
keys=[l.split("\t",1)[1].strip() for l in open("docs/s3_listing.tsv",encoding="utf-8") if l.split("\t",1)[1].startswith(site+"/") and l.strip().lower().endswith(".tif")]
labs=[]
for f in glob.glob(f"data_meta/{site}/**/*.geojson",recursive=True):
    g=gpd.read_file(f).to_crs(utm); g["name"]=os.path.basename(f).split("_",2)[-1][:-8]; labs.append(g)
import pandas as pd; L=pd.concat(labs)
# pick center = centroid of a kurgan cluster
c=L.geometry.centroid; cx,cy=float(np.median(c.x)),float(np.median(c.y))
if len(sys.argv)>3: cx,cy=float(sys.argv[3]),float(sys.argv[4])
fig,axs=plt.subplots(1,len(keys),figsize=(5*len(keys),5.3))
for ax,k in zip(np.atleast_1d(axs),keys):
    with rasterio.open(f"/vsis3/train-expds-3/{k}") as r:
        w=from_bounds(cx-half,cy-half,cx+half,cy+half,r.transform).round_offsets().round_lengths()
        a=r.read(window=w,boundless=True,fill_value=0)
        info=f"{r.count}b {r.dtypes[0]} res={r.res[0]:.3f} crs={'LOCAL' if (r.crs is None or not r.crs.is_epsg_code) else r.crs.to_epsg()}"
    img=np.moveaxis(a[:3],0,-1) if a.shape[0]>=3 else a[0]
    if img.dtype!=np.uint8: lo,hi=np.percentile(img,[1,99]); img=np.clip((img-lo)/(hi-lo+1e-9),0,1)
    ax.imshow(img,extent=(cx-half,cx+half,cy-half,cy+half),cmap="gray")
    for nm,col in zip(sorted(L.name.unique()),["r","y","c","m","lime","orange"]):
        L[L.name==nm].boundary.plot(ax=ax,color=col,lw=0.8)
    ax.set_xlim(cx-half,cx+half); ax.set_ylim(cy-half,cy+half); ax.set_title(os.path.basename(k)[-28:]+"\n"+info,fontsize=8); ax.axis("off")
print(sorted(L.name.unique()), "center",cx,cy)
plt.tight_layout(); plt.savefig(f"viz/{site}.png",dpi=70)
