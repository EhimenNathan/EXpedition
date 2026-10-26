"""Satellite (SpOr) preparation: every scene -> metric UTM grid at a fixed GSD, 8-bit RGB + validity, memmap-ready.

    python scripts/prep_sp.py [--root data_sp] [--gsd 0.5] [--out work/sp]

* CRS: the file CRS when it is a projected UTM CRS; otherwise (EPSG:3857, geographic, missing) the UTM zone of the
  scene centre. Labels are reprojected into the same CRS (they come as 3857 / CRS84 / UTM).
* Resampling: area-average when the native pixel is finer than the target, bilinear otherwise.
* Radiometry: uint8 scenes are used as is (bands 1-3 = RGB by colour interpretation); 16-bit scenes are stretched per
  band to the 1-99% range of valid pixels. Validity = alpha > 0 when present, else any band non-zero.
* Output per scene: <out>/img/<site>__<k>.npy  uint8 (4, H, W) = R, G, B, valid*255; <out>/scenes.csv (transform, CRS,
  native GSD, size); <out>/labels.parquet (site, scene, class, WKB geometry in the scene CRS).
The same reader is used at inference (koz3/sp_io.py) so training and test pixels are produced identically."""
import argparse, glob, json, os, re, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import sp_io
from shapely.geometry import box

ap = argparse.ArgumentParser()
ap.add_argument("--labels_only", action="store_true");
ap.add_argument("--root", default="data_sp"); ap.add_argument("--gsd", type=float, default=0.5); ap.add_argument("--out", default="work/sp")
ap.add_argument("--mod", default="SpOr", choices=["SpOr", "Ae", "Or"])  # data type: picks the image folders and label files
ap.add_argument("--tile_m", type=float, default=0)  # > 0: split each WHOLE scene into tiles of at most this size (fine GSDs)
ap.add_argument("--crop_m", type=float, default=0)  # > 0: read only the label extent + this margin (partially annotated sheets, e.g. Ae)
a = ap.parse_args()
os.makedirs(f"{a.out}/img", exist_ok=True)
CLASSES = ["курганы_целые", "курганы_поврежденные", "городища", "фортификации", "архитектуры"]


def labels_for(site_dir, site, crs):
    import geopandas as gpd
    rows = []
    for f in glob.glob(os.path.join(site_dir, "06_*", "*.geojson")):
        b = os.path.basename(f).replace("Ае", "Ae")  # some names use Cyrillic "Ае"
        m = re.match(rf"^{re.escape(re.sub(r'^\d+_', '', site))}_{a.mod}_(.+)\.geojson$", b)  # label files drop the site number
        if not m or m.group(1) not in CLASSES:
            continue
        g = gpd.read_file(f)
        if not len(g):
            continue
        if g.crs is None:
            g = g.set_crs(3857)
        g = g.to_crs(crs); g = g[g.geometry.notna() & ~g.geometry.is_empty]
        for geom in g.geometry:
            for p in (geom.geoms if geom.geom_type.startswith("Multi") else [geom]):
                if p.geom_type == "Polygon" and p.area > 0:
                    rows.append((m.group(1), p))
    return rows


def crop_windows(tif, site_dir, site, k, max_side_m=2700.0):
    """[(scene id, bounds or None)]. Without --crop_m: the whole scene. With it (partially annotated sheets): one window
    per cluster of labels (labels buffered by crop_m, overlaps merged); clusters wider than max_side_m are split into a
    grid of tiles so no window exceeds read_scene's pixel cap. No labels -> no windows (sheet skipped)."""
    if a.labels_only or (a.crop_m <= 0 and a.tile_m <= 0):
        return [(f"{site}__{k}", None)]
    import rasterio
    from shapely.ops import unary_union
    from rasterio.warp import transform_bounds
    with rasterio.open(tif) as src:
        crs = sp_io.target_crs(src)
        sb = transform_bounds(src.crs or "EPSG:3857", crs, *src.bounds, densify_pts=21)
    if a.crop_m <= 0:  # whole-scene tiling (complete labels: keep every negative pixel)
        l, b, r, t = sb; nx = int(np.ceil((r - l) / a.tile_m)); ny = int(np.ceil((t - b) / a.tile_m))
        if nx * ny == 1:
            return [(f"{site}__{k}", None)]
        return [(f"{site}__{k}_t{ix}{iy}", (l + ix * (r - l) / nx, b + iy * (t - b) / ny, l + (ix + 1) * (r - l) / nx, b + (iy + 1) * (t - b) / ny))
                for ix in range(nx) for iy in range(ny)]
    L0 = labels_for(site_dir, site, crs)
    if not L0:
        print("no labels, skipped (crop mode)", tif); return []
    U = unary_union([p.buffer(a.crop_m) for _, p in L0])
    out = []
    for j, c in enumerate(sorted(getattr(U, "geoms", [U]), key=lambda g: g.bounds)):
        l, b, r, t = c.bounds; nx = int(np.ceil((r - l) / max_side_m)); ny = int(np.ceil((t - b) / max_side_m))
        for ix in range(nx):
            for iy in range(ny):
                w = (l + ix * (r - l) / nx, b + iy * (t - b) / ny, l + (ix + 1) * (r - l) / nx, b + (iy + 1) * (t - b) / ny)
                if nx * ny == 1 or c.intersects(box(*w)):
                    out.append((f"{site}__{k}_{j}" + (f"_{ix}{iy}" if nx * ny > 1 else ""), w))
    return out


scenes, labs = [], []
PREV = pd.read_csv(f"{a.out}/scenes.csv") if a.labels_only else None
t0 = time.time()
for site_dir in sorted(glob.glob(os.path.join(a.root, "*"))):
    site = os.path.basename(site_dir)
    tifs = sorted(t for t in glob.glob(os.path.join(site_dir, "*", "*.tif"))
                  if f"_{a.mod}_" in os.path.basename(os.path.dirname(t)).replace("Ае", "Ae") + "_")
    for k, tif in enumerate(tifs):
      for sid, win in crop_windows(tif, site_dir, site, k):
        outp = f"{a.out}/img/{sid}.npy"
        from shapely.geometry import box
        if a.labels_only:  # images already prepared: reuse their grid from scenes.csv
            r0 = PREV[PREV.scene == sid]
            if not len(r0):
                continue
            r0 = r0.iloc[0]; from rasterio.transform import Affine; from rasterio.crs import CRS
            T = Affine(*json.loads(r0["T"])); crs = CRS.from_user_input(r0["crs"]); H, W = int(r0.H), int(r0.W)
            info = {k_: r0[k_] for k_ in ("native_gsd", "dtype", "bands")}; arr = None
        else:
            try:
                arr, T, crs, info = sp_io.read_scene(tif, a.gsd, bounds=win)
            except Exception as e:
                print("FAIL", tif, repr(e)); continue
            np.save(outp, arr)
            H, W = arr.shape[1:]
        fp = box(T.c, T.f + T.e * H, T.c + T.a * W, T.f)
        n_in = 0
        for cls, p in labels_for(site_dir, site, crs):
            if p.intersects(fp):
                labs.append(dict(site=site, scene=sid, cls=cls, wkb=p.wkb)); n_in += 1
        scenes.append(dict(site=site, scene=sid, file=os.path.relpath(tif, a.root), crs=str(crs), H=H, W=W,
                           T=json.dumps(list(T)[:6]), valid_frac=(round(float((arr[3] > 0).mean()), 3) if arr is not None else float(PREV[PREV.scene == sid].iloc[0].valid_frac)),
                           n_labels=n_in, **info))
        print(f"{sid}: {W}x{H} @ {a.gsd} m from {info['native_gsd']:.3f} m {info['dtype']} {info['bands']}b, {n_in} labels, "
              f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(scenes).to_csv(f"{a.out}/scenes.csv", index=False)
pd.DataFrame(labs).to_parquet(f"{a.out}/labels.parquet")
print("done", len(scenes), "scenes", len(labs), "labels")
