#!/usr/bin/env python3
"""Expand rehearsal optical GT so each source image owns its reference objects."""
from pathlib import Path
import glob, json, sys
import geopandas as gpd
import rasterio
from shapely.geometry import box, mapping

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SRC=ROOT/"work/rehearsal_full/gt.geojson"
OUT=ROOT/"work/rehearsal_full/gt_optical_layers.geojson"
names=json.load(open(ROOT/"work/rehearsal_full/names.json",encoding="utf-8"))
reverse={v:k for k,v in names.items()}
labels=gpd.read_parquet(ROOT/"data/labels_3857.parquet")
base=json.load(open(SRC,encoding="utf-8"))

# Keep other GT unchanged. Rebuild satellite GT from every source image. Ae is
# disabled in v19b and Or has a single validation image, so changing either here
# would add an unrelated difference to the paired satellite experiment.
opt={"SpOr":("sp_or","04_*/*.tif")}
markups={v[0] for v in opt.values()}
features=[f for f in base["features"] if f["properties"].get("markup_type") not in markups]
counts={}
for mod,(markup,pat) in opt.items():
    q=labels[(labels["mod"]==mod)&labels.region.isin(reverse)]
    n=0
    for region,g in q.groupby("region"):
        for path in sorted(glob.glob(str(ROOT/f"data/train-expds-3/{region}/{pat}"))):
            with rasterio.open(path) as src:
                if src.crs is None: continue
                local=g.to_crs(src.crs); keep=local.geometry.representative_point().map(box(*src.bounds).covers).to_numpy()
            for r in g.loc[keep].itertuples():
                features.append({"type":"Feature","geometry":mapping(r.geometry),"properties":{
                    "class_name":r.class_name,"region_name":reverse[region],"markup_type":markup,
                    "confidence":1.0,"image":Path(path).name}})
                n+=1
    counts[markup]=n
out={**base,"features":features}
json.dump(out,open(OUT,"w",encoding="utf-8"),ensure_ascii=False)
print("base",len(base["features"]),"layered",len(features),counts,OUT)
