import geopandas as gpd, glob, os, re, json, collections, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
recs=[]; utm={}
for site in sorted(os.listdir("data_meta")):
    u=os.path.join("data_meta",site,"UTM.json")
    if os.path.exists(u): utm[site]=json.load(open(u,encoding="utf-8"))
    for f in glob.glob(os.path.join("data_meta",site,"**","*.geojson"),recursive=True):
        base=os.path.splitext(os.path.basename(f))[0].replace("Ае","Ae")
        m=re.search(r"_(Li|Ae|SpOr|Or)_(.+)$",base)
        mod,cls=(m.group(1),m.group(2)) if m else ("?",base)
        try: g=gpd.read_file(f)
        except Exception as e: print("ERR",f,e); continue
        raw=json.load(open(f,encoding="utf-8"))
        crs=raw.get("crs",{}).get("properties",{}).get("name")
        gm=g.to_crs(utm[site]["crs"].replace("urn:ogc:def:crs:EPSG::","EPSG:")) if (site in utm and g.crs is not None and len(g)) else g
        for i,r in gm.iterrows():
            geom=r.geometry
            recs.append(dict(site=site,mod=mod,cls=cls,gtype=geom.geom_type if geom is not None else None,
               area=geom.area if geom is not None else np.nan, props=json.dumps({k:v for k,v in g.iloc[i].drop("geometry").items() if pd.notna(v)},ensure_ascii=False,default=str)[:200], crs=crs))
df=pd.DataFrame(recs); df.to_parquet("docs/labels.parquet")
print("UTM example:",list(utm.items())[:2])
print("label CRS:",df.crs.value_counts(dropna=False).to_dict())
print(df.groupby(["cls","gtype"]).size().unstack(fill_value=0))
print(df.groupby(["mod","cls"]).agg(n=("area","size"),sites=("site","nunique"),med_area=("area","median"),p10=("area",lambda x:x.quantile(.1)),p90=("area",lambda x:x.quantile(.9))).round(0))
print("objects per site:",df.groupby("site").size().describe().round(1).to_dict())
print("property samples:"); print(df.drop_duplicates("props").props.head(15).tolist())
