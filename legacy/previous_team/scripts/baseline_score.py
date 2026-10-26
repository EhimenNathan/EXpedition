"""Score the organisers' baseline outputs (scripts/baseline_eval.py) with the OFFICIAL metric (koz3-1.5.0).

    python scripts/baseline_score.py --tag sp  --src data_sp  --mod SpOr     # satellite
    python scripts/baseline_score.py --tag las --src data_meta --mod Li      # LiDAR (labels from data_meta)

GT = the <site>_<mod>_<class>.geojson label files of every region whose baseline run finished with rc=0, reprojected
to EPSG:3857 (region_name = folder name, the same name the baseline writes). Predictions = the regions' result.geojson.
The official F1Calculator does the matching (per region, per class, one-to-one, greedy by confidence).
Writes work/baseline_eval/<tag>/score.json and prints the per-class table."""
import argparse, glob, json, os, re, sys
import geopandas as gpd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3.official.compute_metrics import F1Calculator, normalize_class_name

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="sp"); ap.add_argument("--src", default="data_sp")
ap.add_argument("--mod", default="SpOr"); a = ap.parse_args()
OUT = f"work/baseline_eval/{a.tag}"
pred, gt, regions, failed = [], [], [], []
for sec in sorted(glob.glob(f"{OUT}/*/seconds.txt")):
    reg = os.path.basename(os.path.dirname(sec))
    if "rc=0" not in open(sec).read() or not os.path.exists(f"{OUT}/{reg}/result.geojson"):
        failed.append(reg); continue
    labels = [f for f in glob.glob(f"{a.src}/{reg}/**/*.geojson", recursive=True)
              if re.search(rf"_(?:{a.mod})_", os.path.basename(f).replace("Ае", "Ae"))]
    if not labels:
        continue  # region without labels of this data type: not scored
    regions.append(reg)
    for f in labels:
        cls = normalize_class_name(os.path.basename(f).rsplit(f"_{a.mod}_", 1)[-1][:-8])
        g = gpd.read_file(f)
        if g.crs is None:
            g = g.set_crs(4326)
        for geom in g.to_crs(3857).geometry:
            if geom is not None and not geom.is_empty:
                gt.append({"type": "Feature", "geometry": geom.__geo_interface__, "properties": {"class_name": cls, "region_name": reg}})
    for ft in json.load(open(f"{OUT}/{reg}/result.geojson", encoding="utf-8"))["features"]:
        ft["properties"]["region_name"] = reg; pred.append(ft)
res = F1Calculator().calculate_f1({"type": "FeatureCollection", "features": pred}, {"type": "FeatureCollection", "features": gt})
rows = {c: {k: r[k] for k in r if k in ("num_ground_truth", "num_predictions", "tp", "fp", "fn", "precision", "recall", "f1")}
        for c, r in res["per_class"].items()}
out = dict(final_score=res["final_score"], regions_scored=len(regions), regions_failed=failed, n_pred=len(pred), n_gt=len(gt), per_class=rows)
json.dump(out, open(f"{OUT}/score.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
