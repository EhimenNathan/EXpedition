"""Geophysics data units for KOZ-3 (regions 200-217 in training; region_* folders in validation).

A *unit* is one geophysical image and everything that belongs to it:
    method   MagGr (magnetometry) | EtGr (electrical resistivity) | TerFl (aerial thermography)
    src      the image key after the method: '' for the default MagGr survey (grey map 'серый' and its colour twin
             'цвет' show the same survey), 'цвет_2' (a second survey), 'карта_04_0.75' (resistivity plan map, depth
             slice), 'профиль_14' (resistivity vertical section placed in map coordinates), '' for TerFl
    kind     'map' or 'profile'
    labels   <site>_<method>[_<src>]_FindsPoints.geojson / _ObjectPoly.geojson in 06_*_разметка
    boundary extra/<site>_<method>[_<src>]_границы.geojson (survey outline; predictions must stay inside)
    noise    extra/<site>_<method>[_<src>]_шум.geojson (areas the annotators marked as noise)
Pairing rule found by inspection (2026-10-15): label/boundary files share the image's `src`; the default MagGr labels
(no src) belong to both 'серый' and 'цвет' maps, and the grey map is preferred when both exist."""
import glob, os, re, json, warnings
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling

warnings.filterwarnings("ignore", category=RuntimeWarning)
IMG_RE = re.compile(r"^\d+_(?P<site>.+?)_(?P<method>MagGr|EtGr|TerFl)(?:_(?P<src>.+?))?\.tiff?$")


def _src_key(method, src):
    """Map an image src to its label/boundary key: grey/colour MagGr maps share the default key ''."""
    if method == "MagGr" and src in (None, "", "серый", "цвет"):
        return ""
    return src or ""


def discover(root, prefer_grey=True):
    """List geophysics units under a dataset root (training or validation layout)."""
    units = []
    for reg in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(reg):
            continue
        region = os.path.basename(reg)
        imgs = sorted(glob.glob(os.path.join(reg, "0*_*", "*.tif")) + glob.glob(os.path.join(reg, "0*_*", "*.tiff")))
        by_key = {}
        for t in imgs:
            m = IMG_RE.match(os.path.basename(t))
            if not m:
                continue
            method, src = m.group("method"), m.group("src") or ""
            key = (method, _src_key(method, src))
            by_key.setdefault(key, []).append((src, t))
        for (method, key), lst in by_key.items():
            # one image per key: grey before colour for the default MagGr survey
            lst.sort(key=lambda s: (0 if s[0] in ("серый", "") else 1, s[0]))
            src, path = lst[0]
            # label/boundary stems use the folder's site name (region without its number); the image name can differ
            # (207: image '..._уч7_MagGr_серый.tif', labels 'Семибратнее_Городище_7_MagGr_*'), so try both
            site = re.sub(r"^\d+_", "", region)
            site_img = IMG_RE.match(os.path.basename(path)).group("site")
            stem = f"{site}_{method}" + (f"_{key}" if key else "")
            stem_img = f"{site_img}_{method}" + (f"_{key}" if key else "")
            lab = {k: glob.glob(os.path.join(reg, "06_*", f"{stem}_{k}.geojson")) or glob.glob(os.path.join(reg, "06_*", f"{stem_img}_{k}.geojson"))
                   for k in ("FindsPoints", "ObjectPoly")}
            ex = lambda kind: next(iter(glob.glob(os.path.join(reg, "extra", f"{stem}_{kind}.geojson")) +
                                        glob.glob(os.path.join(reg, "extra", f"{stem_img}_{kind}.geojson"))), None)
            units.append(dict(region=region, site=site, method=method, key=key, src=src, path=path,
                              alt_paths=[p for _, p in lst[1:]],
                              kind="profile" if key.startswith("профиль") else "map",
                              finds=lab["FindsPoints"][0] if lab["FindsPoints"] else None,
                              objects=lab["ObjectPoly"][0] if lab["ObjectPoly"] else None,
                              boundary=ex("границы"), noise=ex("шум")))
    return units


def _read_geoms(path, crs):
    import geopandas as gpd
    if not path or not os.path.exists(path):
        return []
    d = gpd.read_file(path)
    if not len(d):
        return []
    if d.crs is None:
        d = d.set_crs(crs)
    d = d.to_crs(crs)
    d = d[d.geometry.notna() & ~d.geometry.is_empty]  # organisers: empty labels are excluded from validation
    out = []
    for g in d.geometry:
        if g.geom_type == "MultiPolygon":
            out += [p for p in g.geoms if p.area > 0]
        elif g.geom_type == "MultiPoint":
            out += list(g.geoms)
        elif g.geom_type in ("Polygon", "Point"):
            out.append(g)
    return out


def load_unit(u, gsd=0.1):
    """Read the unit image averaged to `gsd` metres; returns dict with float channels, masks, transform, CRS, labels."""
    with rasterio.open(u["path"]) as r:
        f = max(1, int(round(gsd / abs(r.transform.a))))
        H, W = max(1, r.height // f), max(1, r.width // f)
        if r.width * r.height * r.count <= 60_000_000:
            a = r.read(out_shape=(r.count, H, W), resampling=Resampling.average).astype(np.float32)
        else:  # large map (e.g. 200: 3 cm, 0.4 GB): average f x f blocks strip by strip to bound memory
            from rasterio.windows import Window
            a = np.zeros((r.count, H, W), np.float32); rows = max(f, (8_000_000 // max(r.width * r.count, 1)) // f * f)
            for y0 in range(0, H * f, rows):
                h = min(rows, H * f - y0)
                blk = r.read(window=Window(0, y0, W * f, h)).astype(np.float32)
                a[:, y0 // f:(y0 + h) // f] = blk.reshape(r.count, h // f, f, W, f).mean((2, 4))
        T = r.transform * r.transform.scale(r.width / W, r.height / H); crs = r.crs
    rgb = a[:3] if a.shape[0] >= 3 else np.repeat(a[:1], 3, 0)
    lum = rgb.mean(0)
    valid = np.ones(lum.shape, bool)
    if a.shape[0] == 4:
        valid &= a[3] > 0
    # fill colour outside the survey: pure black or pure white
    valid &= ~((rgb.min(0) >= 254) | (rgb.max(0) <= 1))
    bnd = _read_geoms(u["boundary"], crs)
    if bnd:
        valid &= features.rasterize([(g, 1) for g in bnd if g.geom_type == "Polygon"], out_shape=lum.shape,
                                    transform=T, all_touched=True).astype(bool)
    noise = np.zeros(lum.shape, bool)
    nz = _read_geoms(u["noise"], crs)
    if nz:
        noise = features.rasterize([(g, 1) for g in nz if g.geom_type == "Polygon"], out_shape=lum.shape,
                                   transform=T, all_touched=True).astype(bool)
    return dict(lum=lum, rgb=rgb, valid=valid, noise=noise, T=T, crs=crs, gsd=abs(T.a), boundary=bnd,
                finds=[g for g in _read_geoms(u["finds"], crs) if g.geom_type == "Point"],
                objects=[g for g in _read_geoms(u["objects"], crs) if g.geom_type == "Polygon"])


if __name__ == "__main__":
    import sys, pandas as pd
    root = sys.argv[1] if len(sys.argv) > 1 else "data_geo"
    U = discover(root)
    rows = []
    for u in U:
        try:
            d = load_unit(u, 0.1)
            rows.append(dict(region=u["region"], method=u["method"], key=u["key"], kind=u["kind"], img=os.path.basename(u["path"]),
                             alt=len(u["alt_paths"]), H=d["lum"].shape[0], W=d["lum"].shape[1], gsd=round(d["gsd"], 3),
                             valid_m2=round(d["valid"].sum() * d["gsd"] ** 2), noise_m2=round(d["noise"].sum() * d["gsd"] ** 2),
                             finds=len(d["finds"]), objects=len(d["objects"]), boundary=bool(d["boundary"])))
        except Exception as e:
            rows.append(dict(region=u["region"], method=u["method"], key=u["key"], img=os.path.basename(u["path"]), err=repr(e)[:80]))
    D = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(D.to_string(index=False)); D.to_csv("docs/geo_units.csv", index=False)
