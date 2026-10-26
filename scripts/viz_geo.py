"""Figures: each geophysics map with its labels (finds = red dots, objects = cyan outlines, bounds = yellow)."""
import glob, os, sys
import numpy as np, rasterio, geopandas as gpd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rasterio.enums import Resampling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
OUT = "figures/data/geophysics"; os.makedirs(OUT, exist_ok=True)
L = gpd.read_parquet("data/labels_3857.parquet")
for f in sorted(glob.glob("data/train-expds-3/2*/0[79]_*/*.tif")) + sorted(glob.glob("data/train-expds-3/2*/08_*карты/*.tif"))[:6]:
    region = f.split("/")[2]
    with rasterio.open(f) as s:
        k = max(1, max(s.width, s.height) // 1600)
        img = s.read(out_shape=(s.count, s.height // k, s.width // k), resampling=Resampling.average)
        b = s.bounds; crs = s.crs
    rgb = np.moveaxis(img[:3], 0, -1)
    lab = L[(L.region == region) & L["mod"].isin(["MagGr", "EtGr", "TerFl"])].to_crs(crs)
    stem = os.path.basename(f)[:-4]
    fig, ax = plt.subplots(figsize=(10, 10 * rgb.shape[0] / rgb.shape[1] + 0.5))
    ax.imshow(rgb, extent=(b.left, b.right, b.bottom, b.top))
    for cls, kw in (("object_poly", dict(facecolor="none", edgecolor="cyan", linewidth=0.8)),):
        g = lab[(lab.class_name == cls) & lab.file.str.contains(stem.split("_", 1)[1].rsplit("_", 1)[0].split("_MagGr")[0], regex=False)]
        if len(g): g.plot(ax=ax, **kw)
    pts = lab[lab.class_name == "finds_points"]
    if len(pts): ax.scatter(pts.geometry.x, pts.geometry.y, s=6, c="red")
    for e in glob.glob(f"data/train-expds-3/{region}/extra/*.geojson"):
        gpd.read_file(e).to_crs(crs).boundary.plot(ax=ax, color="yellow" if "границ" in e else "magenta", linewidth=1)
    ax.set_xlim(b.left, b.right); ax.set_ylim(b.bottom, b.top)
    ax.set_title(f"{stem}  finds={len(pts)}  objects={int((lab.class_name=='object_poly').sum())}", fontsize=9)
    ax.set_axis_off(); fig.tight_layout(); fig.savefig(f"{OUT}/{stem}.png", dpi=90); plt.close(fig)
    print("saved", stem)
