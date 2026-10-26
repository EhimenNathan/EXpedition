"""One LAS file -> 0.25 m ground DTM + intensity + per-label morphometrics (the kaggle/prep_lidar/prep_lidar.py recipe).

    python scripts/prep_lidar_site.py <site> <las path> <out dir>

Writes <out>/<site>/dtm.tif (float32, -9999 = no data), inten.tif (uint8), morph.csv, info.json. When fewer than 5% of
the points are ground (class 2), e.g. sites 037 and 187, the DTM comes from the progressive morphological filter of
kaggle/prep_lidar/prep_ground.py on a 1 m minimum-elevation grid instead of from all points."""
import os, sys, glob, json, time
import numpy as np, pandas as pd, geopandas as gpd, rasterio, laspy
from rasterio.transform import from_origin
from rasterio import features
from scipy import ndimage as ndi
META = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, META)
from koz3 import dtm as D
R = 1.0  # ground-filter grid (m)


def site_utm(site):
    return json.load(open(f"{META}/data_meta/{site}/UTM.json", encoding="utf-8"))["crs"].replace("urn:ogc:def:crs:EPSG::", "EPSG:")


def accumulate(path):
    with laspy.open(path) as f:
        h = f.header
        xmin, ymin = np.floor(h.mins[0]), np.floor(h.mins[1]); xmax, ymax = np.ceil(h.maxs[0]), np.ceil(h.maxs[1])
        n_tot = h.point_count
        dens_hdr = n_tot / max((xmax - xmin) * (ymax - ymin), 1)
        r = D.grid_res_for_density(dens_hdr)
        W = int(np.ceil((xmax - xmin) / r)); H = int(np.ceil((ymax - ymin) / r))
        acc = {k: np.zeros(H * W, np.float64 if k.startswith("sz") else np.float32) for k in ("cg", "szg", "sig", "ca", "sza")}
        cls_count = np.zeros(256, np.int64)
        for ch in f.chunk_iterator(10_000_000):
            x = np.asarray(ch.x); y = np.asarray(ch.y); z = np.asarray(ch.z)
            cl = np.asarray(ch.classification).astype(np.int64); it = np.asarray(ch.intensity).astype(np.float64)
            cls_count += np.bincount(cl, minlength=256)[:256]
            c = ((x - xmin) / r).astype(np.int64); rr = ((ymax - y) / r).astype(np.int64)
            m = (c >= 0) & (c < W) & (rr >= 0) & (rr < H); idx = rr * W + c
            for sel, cn, szn in ((m, "ca", "sza"), (m & (cl == 2), "cg", "szg")):
                ii = idx[sel]
                acc[cn] += np.bincount(ii, minlength=H * W).astype(np.float32)
                acc[szn] += np.bincount(ii, weights=z[sel], minlength=H * W)
                if cn == "cg": acc["sig"] += np.bincount(ii, weights=it[sel], minlength=H * W).astype(np.float32)
    use_g = acc["cg"].sum() > 0.05 * n_tot
    cnt = (acc["cg"] if use_g else acc["ca"]).reshape(H, W); sz = (acc["szg"] if use_g else acc["sza"]).reshape(H, W)
    si = acc["sig"].reshape(H, W) if use_g else np.zeros((H, W), np.float32)
    info = dict(npts=int(n_tot), res=r, dens=float(dens_hdr), ground_frac=float(cls_count[2] / max(n_tot, 1)), used_ground=bool(use_g),
                classes={int(k): int(v) for k, v in enumerate(cls_count) if v})
    return cnt, sz, si, (xmin, ymin, xmax, ymax), r, info


def save_tif(path, arr, crs, x0, y0, res, dtype, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype, crs=crs,
                       transform=from_origin(x0, y0, res, res), compress="deflate", predictor=3 if dtype == "float32" else 2,
                       tiled=True, blockxsize=512, blockysize=512, nodata=nodata, BIGTIFF="IF_SAFER") as o:
        o.write(arr.astype(dtype), 1)


def labels_for(site, crs):
    L = []
    for f in glob.glob(f"{META}/data_meta/{site}/**/*.geojson", recursive=True):
        b = os.path.basename(f).replace("Ае", "Ae")
        mod = next((m for m in ("_Li_", "_Ae_", "_SpOr_", "_Or_") if m in b), "_?_").strip("_")
        cls = b.split(f"_{mod}_")[-1].replace(".geojson", "") if mod != "?" else b
        g = gpd.read_file(f)
        if g.crs is None or not len(g): continue
        g = g.to_crs(crs); g = g[g.geometry.notna() & ~g.geometry.is_empty]
        g["mod"] = mod; g["cls"] = cls; L.append(g[["mod", "cls", "geometry"]])
    return pd.concat(L, ignore_index=True) if L else None


def morph(dtm, valid, x0, y0, L, inten):
    G = D.MODEL_GSD; rows = []
    for i, (geom, mod, cls) in enumerate(zip(L.geometry, L["mod"], L["cls"])):
        minx, miny, maxx, maxy = geom.bounds; pad = 15
        c0 = int((minx - pad - x0) / G); c1 = int((maxx + pad - x0) / G) + 1
        r0 = int((y0 - maxy - pad) / G); r1 = int((y0 - miny + pad) / G) + 1
        if c1 <= 0 or r1 <= 0 or c0 >= dtm.shape[1] or r0 >= dtm.shape[0]:
            continue
        c0, r0 = max(c0, 0), max(r0, 0); c1, r1 = min(c1, dtm.shape[1]), min(r1, dtm.shape[0])
        w = dtm[r0:r1, c0:c1]; v = valid[r0:r1, c0:c1]
        if w.size < 16: continue
        tr = from_origin(x0 + c0 * G, y0 - r0 * G, G, G)
        m = features.rasterize([(geom, 1)], out_shape=w.shape, transform=tr, all_touched=True).astype(bool)
        if m.sum() < 2 or v[m].mean() < 0.5: continue
        ring = ndi.binary_dilation(m, iterations=int(6 / G)) & ~ndi.binary_dilation(m, iterations=int(2 / G))
        base = float(np.median(w[ring])) if ring.sum() > 8 else np.nan
        zz = w[m]; rr, cc = np.nonzero(m); k = int(np.argmax(zz))
        cen = geom.centroid; px = x0 + (c0 + cc[k] + .5) * G; py = y0 - (r0 + rr[k] + .5) * G
        h = float(zz.max() - base)
        l5 = D.lrm(w, 5.0); lg = D.log_blob(w, max(1.0, np.sqrt(geom.area / np.pi)))
        iw = inten[r0:r1, c0:c1]
        rows.append(dict(idx=i, mod=mod, cls=cls, area=geom.area, perim=geom.length, height=h,
                         h_mean=float(zz.mean() - base), lrm5_max=float(l5[m].max()), lrm5_mean=float(l5[m].mean()),
                         log_max=float(lg[m].max()), peak_off=float(np.hypot(px - cen.x, py - cen.y)),
                         iso10_ratio=float(((zz >= base + .1 * h).sum() * G * G) / geom.area) if h > 0 else np.nan,
                         slope_ring=float(D.slope_deg(w)[ring].mean()) if ring.sum() else np.nan,
                         inten_contrast=float(iw[m].mean() - iw[ring].mean()) if ring.sum() else np.nan))
    return pd.DataFrame(rows)



def pmf(zmin, cell=R, s=0.15, dh0=0.3, dhmax=3.0, wins=(3, 5, 9, 17, 33)):
    z = zmin.copy(); ground = np.isfinite(z)
    z = np.where(ground, z, np.nanmax(zmin))
    prev = z.copy(); w_prev = 1
    for w in wins:
        op = ndi.grey_opening(prev, size=(w, w))
        dh = min(dh0 + s * (w - w_prev) * cell, dhmax)
        ground &= (prev - op) <= dh
        prev, w_prev = op, w
    return ground




def ground_dtm(path):
    """PMF ground DTM for LAS files without ground classification (prep_ground.py)."""
    with laspy.open(path) as f:
        h = f.header; xmin, ymin = np.floor(h.mins[0]), np.floor(h.mins[1]); xmax, ymax = np.ceil(h.maxs[0]), np.ceil(h.maxs[1])
        W = int(np.ceil((xmax - xmin) / R)); H = int(np.ceil((ymax - ymin) / R)); zmin = np.full(H * W, np.inf)
        for ch in f.chunk_iterator(10_000_000):
            c = ((np.asarray(ch.x) - xmin) / R).astype(np.int64); r = ((ymax - np.asarray(ch.y)) / R).astype(np.int64)
            m = (c >= 0) & (c < W) & (r >= 0) & (r < H)
            np.minimum.at(zmin, r[m] * W + c[m], np.asarray(ch.z)[m])
    zmin = zmin.reshape(H, W); zmin[~np.isfinite(zmin)] = np.nan
    g = pmf(zmin); cnt = g.astype(np.float32); sz = np.where(g, np.nan_to_num(zmin), 0.0)
    dtm, _ = D._normconv_fill(sz, cnt, max_sigma_px=16)
    valid = D.resample_to((ndi.distance_transform_edt(~g) * R <= 6.0).astype(np.float32), R, order=0) > 0.5
    return D.resample_to(np.nan_to_num(dtm, nan=float(np.nanmedian(dtm))), R), valid, xmin, ymax, float(g.mean())


def main(site, local, out_root):
    t0 = time.time(); crs = site_utm(site); out = f"{out_root}/{site}"; os.makedirs(out, exist_ok=True)
    cnt, sz, si, (xmin, ymin, xmax, ymax), r, info = accumulate(local)
    dtm, hole = D._normconv_fill(sz, cnt, max_sigma_px=int(8 / r)); inten, _ = D._normconv_fill(si, cnt, max_sigma_px=int(8 / r))
    valid_g = ndi.distance_transform_edt(cnt == 0) * r <= 3.0
    dtm = D.resample_to(np.nan_to_num(dtm, nan=np.nanmedian(dtm)), r); inten = D.resample_to(np.nan_to_num(inten), r)
    valid = D.resample_to(valid_g.astype(np.float32), r, order=0) > 0.5
    cov = D.resample_to((cnt > 0).astype(np.float32), r, order=0)
    if not info.get("used_ground", True) or os.environ.get("KOZ3_FORCE_PMF"):  # no ground classification: PMF ground DTM
        gd, gv, info["pmf_ground_frac"] = D.pmf_ground(local, xmin, ymax, *dtm.shape)
        dtm, valid = gd, valid & gv
    lo, hi = np.percentile(inten[valid], [1, 99]) if valid.any() else (0, 1)
    inten8 = np.clip((inten - lo) / max(hi - lo, 1e-6) * 254 + 1, 1, 255); inten8[~valid] = 0
    save_tif(f"{out}/dtm.tif", np.where(valid, dtm, -9999), crs, xmin, ymax, D.MODEL_GSD, "float32", -9999)
    save_tif(f"{out}/inten.tif", inten8, crs, xmin, ymax, D.MODEL_GSD, "uint8", 0)
    L = labels_for(site, crs)
    M = morph(dtm, valid, xmin, ymax, L, inten) if L is not None else pd.DataFrame()
    if len(M): M.insert(0, "site", site); M.to_csv(f"{out}/morph.csv", index=False)
    info.update(site=site, H=dtm.shape[0], W=dtm.shape[1], km2=float(valid.mean() * dtm.size * D.MODEL_GSD ** 2 / 1e6),
                obs_cell_frac=float(cov.mean()), n_morph=len(M), t_total=round(time.time() - t0, 1), crs=crs)
    json.dump(info, open(f"{out}/info.json", "w", encoding="utf-8"), ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in info.items() if k != "classes"}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
