import os, rasterio, pandas as pd, struct, boto3, concurrent.futures as cf
from botocore.config import Config
os.environ.update(AWS_S3_ENDPOINT="s3.ru-7.storage.selcloud.ru",
  AWS_ACCESS_KEY_ID=os.environ["KOZ3_S3_ACCESS_KEY"],
  AWS_SECRET_ACCESS_KEY=os.environ["KOZ3_S3_SECRET_KEY"], AWS_VIRTUAL_HOSTING="FALSE", AWS_REGION="ru-7",
  GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff")
s3 = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru", aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], region_name="ru-7", config=Config(s3={"addressing_style":"path"}))
rows=[l.rstrip("\n").split("\t",1) for l in open("docs/s3_listing.tsv",encoding="utf-8")]
def tif(k,s):
    try:
        with rasterio.open(f"/vsis3/train-expds-3/{k}") as r:
            ov=r.overviews(1) if r.count else []
            return dict(key=k,size=int(s),crs=str(r.crs)[:40] if r.crs else None,resx=r.res[0],resy=r.res[1],w=r.width,h=r.height,bands=r.count,
              dtype=r.dtypes[0],nodata=r.nodata,ci=",".join(c.name for c in r.colorinterp),tiled=r.profile.get("tiled"),compress=str(r.compression),ovr=len(ov),
              b=r.bounds)
    except Exception as e: return dict(key=k,size=int(s),err=str(e)[:120])
def las(k,s):
    try:
        h=s3.get_object(Bucket="train-expds-3",Key=k,Range="bytes=0-374")["Body"].read()
        vmaj,vmin=h[24],h[25]; fmt=h[104]; reclen=struct.unpack("<H",h[105:107])[0]
        legacy=struct.unpack("<I",h[107:111])[0]
        n=struct.unpack("<Q",h[247:255])[0] if (vmin>=4 and len(h)>=255) else legacy
        mx,mnx,my,mny,mz,mnz=struct.unpack("<6d",h[179:227])
        return dict(key=k,size=int(s),las_ver=f"{vmaj}.{vmin}",pfmt=fmt,npts=n,area=(mx-mnx)*(my-mny),dens=n/max((mx-mnx)*(my-mny),1),zr=mz-mnz,minx=mnx,miny=mny)
    except Exception as e: return dict(key=k,size=int(s),err=str(e)[:120])
jobs=[(k,s) for s,k in rows]
with cf.ThreadPoolExecutor(16) as ex:
    T=list(ex.map(lambda a: tif(*a), [j for j in jobs if j[0].lower().endswith((".tif",".tiff"))]))
    L=list(ex.map(lambda a: las(*a), [j for j in jobs if j[0].lower().endswith(".las")]))
pd.DataFrame(T).to_csv("docs/tif_headers.csv",index=False); pd.DataFrame(L).to_csv("docs/las_headers.csv",index=False)
print(len(T),len(L))
