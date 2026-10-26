"""Validation-shaped input from training regions (data only, anonymised region_XXXX, no labels) + its GT.

  python scripts/make_val_copy.py OUT_DIR region_prefix [region_prefix ...]
Writes OUT_DIR/input/region_XXXX/... (symlinks to the data files, extra/ bounds+noise, UTM.json) and
OUT_DIR/gt.geojson (labels of those regions in the official GT format) and OUT_DIR/names.json.
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3.labels import DATA, to_gt_geojson  # noqa: E402


def main():
    out, prefixes = sys.argv[1], sys.argv[2:]
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    names = {}
    for pre in prefixes:
        for rd in sorted(glob.glob(os.path.join(DATA, pre + "*"))):
            region = os.path.basename(rd)
            anon = "region_" + hashlib.md5(region.encode()).hexdigest()[:4].upper()
            names[anon] = region
            dst = os.path.join(out, "input", anon)
            for f in glob.glob(os.path.join(rd, "0[1-9]_*", "*")):
                if "_разметка" in f or "_Li_" in os.path.basename(f):
                    continue
                folder = os.path.basename(os.path.dirname(f)).replace(region.split("_", 1)[1], anon)
                fn = os.path.basename(f).replace(region.split("_", 1)[1], anon)
                os.makedirs(os.path.join(dst, folder), exist_ok=True)
                link = os.path.join(dst, folder, fn)
                if not os.path.exists(link):
                    os.symlink(os.path.abspath(f), link)
            for f in glob.glob(os.path.join(rd, "extra", "*")):
                os.makedirs(os.path.join(dst, "extra"), exist_ok=True)
                link = os.path.join(dst, "extra", os.path.basename(f).replace(region.split("_", 1)[1], anon))
                if not os.path.exists(link):
                    os.symlink(os.path.abspath(f), link)
            if os.path.exists(os.path.join(rd, "UTM.json")):
                link = os.path.join(dst, "UTM.json")
                if not os.path.exists(link):
                    os.symlink(os.path.abspath(os.path.join(rd, "UTM.json")), link)
    sub = L[L.region.isin(names.values())].copy()
    inv = {v: k for k, v in names.items()}
    sub["region"] = sub.region.map(inv)
    json.dump(to_gt_geojson(sub), open(os.path.join(out, "gt.geojson"), "w"), ensure_ascii=False)
    json.dump(names, open(os.path.join(out, "names.json"), "w"), ensure_ascii=False, indent=1)
    print(len(names), "regions;", len(sub), "GT objects")


if __name__ == "__main__":
    main()
