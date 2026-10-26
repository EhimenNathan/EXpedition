"""Shared inference helpers: logging, region / data-type discovery, CRS, geometry → output features, atomic write."""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury",
           "fortifikatsii", "finds_points", "object_poly")
MAX_FEATURES = 19500       # platform limit 20,000
MAX_VERTICES = 2900        # platform limit 3,000 per polygon

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def resolve_output_path(arg):
    p = Path(arg)
    if p.suffix.lower() == ".geojson":
        return p
    return p / "result.geojson"


def write_result_atomic(features, out_path):
    """Write once: temp file in the same directory → fsync → os.replace (atomic on one filesystem)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features = sorted(features, key=lambda f: -f["properties"].get("confidence", 0.0))[:MAX_FEATURES]
    tmp = out_path.parent / ("temp_" + out_path.name)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


# --- region and data-type discovery ---------------------------------------------------------------

# folder / file tag → (markup_type, data family)
TAGS = [
    (r"_La_", "li", "las"),
    (r"_Li_", "li", "li"),
    (r"_(Ae|Ае)_", "ae", "ae"),
    (r"_SpOr_", "sp_or", "sp_or"),
    (r"_Or_", "or", "or"),
    (r"_MagGr_", "maggr", "maggr"),
    (r"_EtGr_", "etgr", "etgr"),
    (r"_TerFl", "terfl", "terfl"),
]


def region_dirs(in_dir):
    in_dir = Path(in_dir)
    dirs = sorted(d for d in in_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    return dirs or [in_dir]


def region_files(region):
    """{family: [paths]} for the rasters / point clouds of one region (labels and extra/ excluded)."""
    out = {}
    for p in sorted(Path(region).rglob("*")):
        if not p.is_file() or "extra" in p.parts or "разметка" in str(p.parent.name):
            continue
        if p.suffix.lower() not in (".tif", ".tiff", ".las", ".laz"):
            continue
        name = "_" + p.name
        for pat, _mt, fam in TAGS:
            if re.search(pat, name) or re.search(pat, "_" + p.parent.name + "_"):
                if fam == "li" and p.suffix.lower() in (".las", ".laz"):
                    fam = "las"
                out.setdefault(fam, []).append(p)
                break
    return out


def region_extra(region, kind):
    """extra/*_<kind>.geojson polygons (kind: 'границы' = survey bounds, 'шум' = noise) as shapely in EPSG:3857."""
    import geopandas as gpd
    polys = []
    for p in sorted((Path(region) / "extra").glob(f"*_{kind}.geojson")):
        try:
            g = gpd.read_file(p)
            if g.crs is None:
                g = g.set_crs(utm_crs(region))
            polys.append((p.name, g.to_crs(3857).union_all()))
        except Exception as e:
            log(f"extra read failed {p}: {e}")
    return polys


def utm_crs(region, sample_lonlat=None):
    """CRS string from UTM.json; if the stated zone disagrees with the data longitude by > 6°, derive it."""
    f = Path(region) / "UTM.json"
    crs = None
    if f.exists():
        try:
            d = json.load(open(f, encoding="utf-8"))
            crs = d.get("crs")
            if not crs and d.get("utm"):
                z = str(d["utm"]).upper()
                crs = f"EPSG:{(32600 if z.endswith('N') else 32700) + int(re.sub('[^0-9]', '', z))}"
        except Exception as e:
            log(f"UTM.json unreadable in {region}: {e}")
    if crs and sample_lonlat is not None:
        m = re.search(r"(\d{5})$", crs)
        if m:
            zone = int(m.group(1)) % 100
            lon0 = -183 + 6 * zone
            if abs(sample_lonlat[0] - lon0) > 6:
                crs = f"EPSG:{32600 + int((sample_lonlat[0] + 180) // 6) + 1}"
    return crs


def raster_crs(src, region):
    """Raster CRS, falling back to UTM.json when the file has none or only a LOCAL_CS."""
    try:
        if src.crs is not None and src.crs.to_epsg() is not None:
            return src.crs.to_string()
    except Exception:
        pass
    return utm_crs(region)


# --- geometry → features ---------------------------------------------------------------------------

def to_feature(geom_3857, cls, conf, region_name, markup_type):
    from shapely.geometry import mapping
    from shapely import set_precision
    if geom_3857.geom_type != "Polygon":
        return None
    if len(geom_3857.exterior.coords) > MAX_VERTICES:
        tol = 0.05
        while len(geom_3857.exterior.coords) > MAX_VERTICES:
            geom_3857 = geom_3857.simplify(tol); tol *= 2
    geom_3857 = set_precision(geom_3857, 0.001)
    if geom_3857.is_empty or not geom_3857.is_valid:
        return None
    return {"type": "Feature", "geometry": mapping(geom_3857),
            "properties": {"class_name": cls, "region_name": region_name, "sub_region_name": "",
                           "markup_type": markup_type, "confidence": float(round(conf, 5)),
                           "crs": "urn:ogc:def:crs:EPSG::3857"}}


def circle(x, y, r, n=24):
    from shapely.geometry import Polygon
    return Polygon([(x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n)) for i in range(n)])


def oom_retry(fn, tiles=(1024, 512, 256)):
    """Run fn(tile) and retry with smaller tiles on CUDA out-of-memory (the platform GPU is free, but huge sites
    or a shared GPU must not cost a whole region)."""
    import torch
    for i, t in enumerate(tiles):
        try:
            return fn(t)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"CUDA OOM at tile {t}" + (f", retrying at {tiles[i + 1]}" if i + 1 < len(tiles) else ", giving up"))
            if i + 1 == len(tiles):
                raise
