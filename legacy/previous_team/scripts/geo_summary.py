"""Summarise the KOZ-3 geophysics regions (200-217): rasters per method and labels per class/method."""
import glob, os, re, sys, collections
import rasterio, geopandas as gpd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data_geo"
METH = re.compile(r"_(MagGr|EtGr|TerFl)_")
for reg in sorted(glob.glob(os.path.join(ROOT, "2*"))):
    print("\n##", os.path.basename(reg), open(os.path.join(reg, "UTM.json"), encoding="utf-8").read().strip())
    for t in sorted(glob.glob(os.path.join(reg, "0*", "*.tif"))):
        with rasterio.open(t) as r:
            print(f"  tif {os.path.basename(t):60s} {r.count}b {r.dtypes[0]:6s} {r.width}x{r.height} gsd={abs(r.transform.a):.3f} crs={r.crs.to_epsg() if r.crs else None}")
    cnt = collections.Counter()
    for g in sorted(glob.glob(os.path.join(reg, "06_*", "*.geojson"))):
        m = METH.search(os.path.basename(g)); kind = os.path.basename(g).rsplit("_", 1)[-1].replace(".geojson", "")
        try:
            d = gpd.read_file(g); d = d[d.geometry.notna() & ~d.geometry.is_empty]
        except Exception as e:
            print("  ERR", g, e); continue
        key = (m.group(1) if m else "?", kind, ",".join(sorted(set(d.geom_type))))
        cnt[key] += len(d)
    for k, v in sorted(cnt.items()):
        print(f"  labels {k}: {v}")
