"""Training labels: every 06_*_разметка/*.geojson as one GeoDataFrame in EPSG:3857.

Class and data type come from the file name (labels carry no class_name):
  <site>_<Li|Ae|SpOr|Or>_<class>.geojson,  <site>_<MagGr|EtGr|TerFl>[_key]_<FindsPoints|ObjectPoly>.geojson
Empty geometries are dropped (organisers exclude them in validation too).
"""
import glob
import json
import os
import re

import warnings

warnings.filterwarnings("ignore", message=".*GeoSeries.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "train-expds-3")

CLASS_RU = {
    "курганы_поврежденные": "kurgany_povrezhdennye",
    "курганы_целые": "kurgany_tselye",
    "городища": "gorodishcha",
    "архитектуры": "arkhitektury",
    "фортификации": "fortifikatsii",
    "FindsPoints": "finds_points",
    "ObjectPoly": "object_poly",
}
MOD_MARKUP = {"Li": "li", "La": "li", "Ae": "ae", "SpOr": "sp_or", "Or": "or",
              "MagGr": "maggr", "EtGr": "etgr", "TerFl": "terfl"}
_MOD_RE = re.compile(r"_(Li|La|Ae|Ае|SpOr|Or|MagGr|EtGr|TerFl)(?:_|$)")


def parse_name(fname):
    stem = os.path.basename(fname)[:-len(".geojson")]
    m = _MOD_RE.search(stem)
    if not m:
        return None
    mod = m.group(1).replace("Ае", "Ae")
    rest = stem[m.end():]
    cls = next((v for k, v in CLASS_RU.items() if rest.endswith(k)), None)
    key = rest[: -len(next(k for k in CLASS_RU if rest.endswith(k)))].strip("_") if cls else ""
    return mod, cls, key


def region_utm(region_dir):
    f = os.path.join(region_dir, "UTM.json")
    if os.path.exists(f):
        return json.load(open(f, encoding="utf-8")).get("crs")
    return None


def read_any(path, fallback_crs):
    import geopandas as gpd
    g = gpd.read_file(path)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    if len(g) == 0:
        return g.set_crs(3857, allow_override=True) if g.crs is None else g.to_crs(3857)
    if g.crs is None:
        # guess: degrees → CRS84, huge eastings → 3857, else the region's UTM
        x = abs(g.total_bounds[0])
        g = g.set_crs(4326 if x <= 180 else (3857 if x > 1_000_000 else fallback_crs))
    return g.to_crs(3857)


def load_labels(data_dir=DATA, explode=False):
    """explode=False keeps MultiPolygons (the official parser scores their largest part); True for training targets."""
    import geopandas as gpd
    import pandas as pd
    parts = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*", "06_*", "*.geojson"))):
        p = parse_name(f)
        if p is None or p[1] is None:
            print("skip label file", f)
            continue
        mod, cls, key = p
        region = os.path.relpath(f, data_dir).split(os.sep)[0]
        g = read_any(f, region_utm(os.path.join(data_dir, region)))
        if len(g) == 0:
            continue
        g = g[["geometry"]].copy()
        g["region"], g["mod"], g["markup_type"], g["class_name"], g["key"], g["file"] = \
            region, mod, MOD_MARKUP[mod], cls, key, os.path.basename(f)
        parts.append(g)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=3857)
    if explode:
        out = out.explode(index_parts=False).reset_index(drop=True)
    return out


def to_gt_geojson(gdf):
    """GeoJSON dict in the format compute_metrics.py expects for ground truth."""
    feats = []
    for r in gdf.itertuples():
        feats.append({"type": "Feature", "geometry": r.geometry.__geo_interface__,
                      "properties": {"class_name": r.class_name, "region_name": r.region,
                                     "markup_type": r.markup_type, "confidence": 1.0}})
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
            "features": feats}


if __name__ == "__main__":
    L = load_labels()
    L.to_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    print(len(L), "label geometries")
    print(L.groupby(["mod", "class_name"]).agg(n=("region", "size"), regions=("region", "nunique")).to_string())
    print(L.geom_type.value_counts().to_string())
