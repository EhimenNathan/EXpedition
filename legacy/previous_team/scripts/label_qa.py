import geopandas as gpd, pandas as pd, numpy as np, glob, os, re, json, warnings
from shapely.geometry import box
from shapely import make_valid
warnings.filterwarnings("ignore")
T=pd.read_csv("docs/tif_headers.csv")
T["site"]=T.key.str.split("/").str[0]
T["mod"]=T.key.str.extract(r"_(Li|Ae|SpOr|Or)_[^/]+/")[0]
T["bb"]=T.b.str.extract(r"left=([-\d.e]+), bottom=([-\d.e]+), right=([-\d.e]+), top=([-\d.e]+)").astype(float).values.tolist()
rows=[]; qa=[]
for site in sorted(os.listdir("data_meta")):
    utm=json.load(open(f"data_meta/{site}/UTM.json",encoding="utf-8"))["crs"].replace("urn:ogc:def:crs:EPSG::","EPSG:")
    foot={}
    for _,t in T[T.site==site].iterrows():
        crs=t.crs if isinstance(t.crs,str) and t.crs.startswith("EPSG") else utm
        g=gpd.GeoSeries([box(*t.bb)],crs=crs).to_crs(utm).iloc[0]
        foot.setdefault(t["mod"],[]).append(g)
    for f in glob.glob(f"data_meta/{site}/**/*.geojson",recursive=True):
        base=os.path.basename(f).replace("Ае","Ae")
        m=re.search(r"_(Li|Ae|SpOr|Or)_(.+)\.geojson$",base); mod,cls=m.group(1),m.group(2)
        g=gpd.read_file(f)
        if g.crs is None: qa.append((site,base,"NO_CRS")); continue
        g=g.to_crs(utm)
        inv=(~g.is_valid).sum(); emp=g.is_empty.sum()
        g["geometry"]=g.geometry.apply(make_valid)
        for i,geom in enumerate(g.geometry):
            if geom is None or geom.is_empty: continue
            c=geom.representative_point()
            inside={mm:any(fp.contains(c) for fp in fps) for mm,fps in foot.items()}
            rows.append(dict(site=site,mod=mod,cls=cls,area=geom.area,nparts=len(getattr(geom,"geoms",[geom])),
                x=c.x,y=c.y,in_own=inside.get(mod,False),in_any=any(inside.values()),own_has=mod in foot,
                minrect=geom.minimum_rotated_rectangle.area if geom.area>0 else 0,perim=geom.length))
        if inv or emp: qa.append((site,base,f"invalid={inv} empty={emp}"))
D=pd.DataFrame(rows); D.to_parquet("docs/labels_qa.parquet")
print("objects",len(D))
print("label modality has NO raster of that modality:",D[~D.own_has].groupby(["site","mod"]).size().to_dict())
print("centroid outside own-modality footprint:", (~D.in_own & D.own_has).sum(), D[~D.in_own & D.own_has].groupby(["site","mod"]).size().sort_values(ascending=False).head(15).to_dict())
print("outside ANY raster:", (~D.in_any).sum())
print("zero/tiny area (<1 m2):", (D.area<1).sum(), D[D.area<1].groupby(["mod","cls"]).size().to_dict())
print("multipart:", (D.nparts>1).sum())
D["compact"]=4*np.pi*D.area/np.maximum(D.perim,1e-9)**2
print(D.groupby("cls").compact.describe().round(2))
print("QA notes:",qa[:30])
