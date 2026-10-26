"""LiDAR → Ae twins with the Ae expert OFF (validation region 789Q has LAS + German aerial film), nested + robust.

Training: in regions with both Li and Ae labels, 46–52 % of Ae kurgan labels contain a Li label's representative
point (same mound labelled twice; cemeteries 65–71 %). The deployed fuse only twins into a data type that produced
detections → with Ae off, LiDAR→Ae twins vanished. Here: deployed LiDAR OOF decode (blend + prior, thr 0.15, dual
0.05) on sites that also carry Ae labels; GT = Li + Ae labels; the twin set duplicates kurgan detections (classes as
emitted, optionally only the primary class / only conf ≥ c_min) at the same position.
  python scripts/li_ae_twin_oof.py
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main():
    import copy
    import geopandas as gpd
    from shapely.geometry import shape
    from koz3.li_train import labels, site_items, decode_site_v2
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from robust_accept import site_counts, report
    import li_compare_oof as lc
    Lall = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    ae_regions = set(Lall[Lall["mod"] == "Ae"].region)
    sites = sorted(r for r in P["ens"] if r in ae_regions); fold = {r: items[r]["fold"] for r in sites}
    print(len(sites), "LiDAR OOF sites with Ae labels:", sites)
    pri = {"R": 200.0, "K": 3, "snb": 0.2}; D = {}; FP = {}
    import glob, json, rasterio
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import unary_union
    for r in sites:                  # Ae sheet footprints (valid-data bbox of every sheet, EPSG:3857) — known at test time
        bs = []
        for t in glob.glob(os.path.join(ROOT, "data/train-expds-3", r, "*_Ae_*", "*.tif")):
            with rasterio.open(t) as src:
                crs = src.crs.to_string() if (src.crs is not None and src.crs.to_epsg()) else json.load(open(os.path.join(ROOT, "data/train-expds-3", r, "UTM.json")))["crs"]
                tr = Transformer.from_crs(crs, 3857, always_xy=True); b = src.bounds
                xs, ys = tr.transform([b.left, b.right, b.right, b.left], [b.bottom, b.bottom, b.top, b.top])
                bs.append(box(min(xs), min(ys), max(xs), max(ys)))
        FP[r] = unary_union(bs) if bs else None
        print(r, "Ae footprint km²", round(FP[r].area / 1e6, 3) if FP[r] is not None else None)
    for r in sites:
        it = items[r]
        b = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32)
        D[r] = decode_site_v2(b, it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05, comp_thr=0.5, prior=pri)

    def site_fc(st, r):
        fs = list(D[r])
        if st[0] != "base":
            mode, cmin = st
            kf = [f for f in D[r] if f["properties"]["class_name"] in K and f["properties"]["confidence"] >= cmin]
            if mode.startswith("small"):     # only when the Ae coverage is a site crop (footprint ≤ A km²), inside it
                A = float(mode.split("_")[1])
                if FP[r] is None or FP[r].area / 1e6 > A:
                    kf = []
                kf = [f for f in kf if FP[r].contains(shape(f["geometry"]).centroid)]
            if mode.endswith("_fp"):         # only inside the Ae sheet footprint
                kf = [f for f in kf if FP[r] is not None and FP[r].contains(shape(f["geometry"]).centroid)]
            if mode.startswith("primary"):            # one twin per location: the highest-confidence class
                best = {}
                for f in kf:
                    k = tuple(np.round(shape(f["geometry"]).centroid.coords[0], 1))
                    if k not in best or f["properties"]["confidence"] > best[k]["properties"]["confidence"]:
                        best[k] = f
                kf = list(best.values())
            for f in kf:
                t = copy.deepcopy(f); t["properties"]["markup_type"] = "ae"; t["properties"]["confidence"] *= 0.1; fs.append(t)
        return {"type": "FeatureCollection", "features": fs}
    G = Lall[Lall.region.isin(sites) & Lall["mod"].isin(["Li", "Ae"])]
    folds = sorted(set(fold.values()))
    gt = {f: to_gt_geojson(G[G.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    SS = [("base",)] + [(m, c) for m in ("all_fp", "small_1", "small_2", "small_6") for c in (0.15, 0.25, 0.4)]
    C = fold_counts(lambda st, f: {"type": "FeatureCollection", "features": [g for r in sites if fold[r] == f for g in site_fc(st, r)["features"]]}, SS, folds, gt)
    for st in SS:
        tot = _add_all(C[st].values())
        print(f"{st}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  kt {tuple(int(v) for v in tot[K[1]])}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"no twins {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(G[G.region == r]) for r in sites}
    CS = site_counts(site_fc, (SS[0], os_), sites, gtr)
    report(f"LiDAR→Ae twins {os_}", CS[SS[0]], CS[os_])
    for st in (("small_6", 0.15), ("small_6", 0.25), ("small_6", 0.4), ("small_2", 0.25)):
        if st != os_:
            CS = site_counts(site_fc, (SS[0], st), sites, gtr); report(f"LiDAR→Ae twins {st}", CS[SS[0]], CS[st])


if __name__ == "__main__":
    main()
