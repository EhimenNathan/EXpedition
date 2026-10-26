"""Organisers' baseline (YOLO11s-seg) vs our A2 LiDAR out-of-fold predictions, same sites, official metric (koz3-1.5.0).

    python scripts/compare_baseline_las.py [--tag A2_ref_unet_r34_all]

Baseline: work/baseline_eval/las/<site>/result.geojson (kaggle/sola/k_baseline_las.py), GT = the Li label files in
data_meta. A2: kaggle/out/gpu_loss/<tag> OOF, scored with kaggle/official_rescore.py at its best single (mask, conf)
setting on the same pool. Pools: all common sites, and the sites new in KOZ-3 (>= 179), which the KOZ-2 baseline cannot
have seen in training (sites 001-069 may be in-sample for it)."""
import argparse, glob, json, os, re, sys
import geopandas as gpd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "kaggle"))
from koz3.official.compute_metrics import F1Calculator, normalize_class_name
import official_rescore as R

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="A2_ref_unet_r34_all"); a = ap.parse_args()
BL = os.path.join(ROOT, "work", "baseline_eval", "las")


def baseline(sites):
    pred, gt = [], []
    for s in sites:
        for f in glob.glob(f"{ROOT}/data_meta/{s}/**/*_Li_*.geojson", recursive=True):
            cls = normalize_class_name(os.path.basename(f).rsplit("_Li_", 1)[-1][:-8]); g = gpd.read_file(f)
            g = g.set_crs(4326) if g.crs is None else g
            gt += [{"type": "Feature", "geometry": x.__geo_interface__, "properties": {"class_name": cls, "region_name": s}}
                   for x in g.to_crs(3857).geometry if x is not None and not x.is_empty]
        for ft in json.load(open(f"{BL}/{s}/result.geojson", encoding="utf-8"))["features"]:
            ft["properties"]["region_name"] = s; pred.append(ft)
    r = F1Calculator().calculate_f1({"type": "FeatureCollection", "features": pred}, {"type": "FeatureCollection", "features": gt})
    return r["final_score"], {c: round(v["f1"], 3) for c, v in r["per_class"].items() if v["num_ground_truth"]}


def ours(P, G):
    best = None
    for m in sorted(P.thr.unique()):
        for c in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65):
            f, rows = R.score(P[P.thr == m], G, c)
            if best is None or f > best[0]:
                best = (f, m, c, {k: round(v["F1"], 3) for k, v in rows.items() if v["n_gt"]})
    return best


P, G = R.load(a.tag)
bl_sites = {os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{BL}/*/result.geojson")}
common = sorted(bl_sites & set(G.group))
print(f"A2 OOF sites {G.group.nunique()}, baseline sites {len(bl_sites)}, common {len(common)}; "
      f"baseline-only: {sorted(bl_sites - set(G.group))}")
for name, pool in (("all common", common), ("new in KOZ-3 (>=179)", [s for s in common if int(s[:3]) >= 179])):
    fb, cb = baseline(pool)
    fo, m, c, co = ours(P[P.group.isin(pool)], G[G.group.isin(pool)])
    print(f"\n## {name}: {len(pool)} sites\n  baseline final {fb:.4f}  {cb}\n  ours {a.tag} final {fo:.4f} (mask {m}, conf>={c})  {co}")
