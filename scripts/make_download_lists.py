"""Write signed-URL download lists (URL TAB key TAB size, valid 7 days) per data type from the latest S3 snapshot.

Usage: python scripts/make_download_lists.py ; then bash docs/download_lists/download.sh <list> data
"""
import csv, glob, os, re
from s3_changes import client, BUCKET

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "download_lists")


def kind(key):
    n = key.split("/")[-1]
    if key.endswith("/"): return None
    if not n.lower().endswith((".tif", ".tiff", ".las", ".laz")): return "0_labels_meta"
    if re.search(r"_Li_", n): return "6_lidar_maps_Li"
    if re.search(r"_Or_", n): return "5_orthophoto_Or"
    if n.lower().endswith((".las", ".laz")): return "4_lidar_LAS"
    if re.search(r"_(Ae|Ае)_", n): return "3_aerial_Ae"
    if "SpOr" in n: return "2_satellite_SpOr"
    if re.search(r"(MagGr|EtGr|TerFl)", n): return "1_geophysics"
    return "0_labels_meta"


def main():
    snap = sorted(glob.glob(os.path.join(ROOT, "docs", "s3_snapshots", "snapshot_*.csv")))[-1]
    s3 = client(); lists = {}
    for r in csv.DictReader(open(snap, encoding="utf-8")):
        k = kind(r["key"])
        if k is None: continue
        url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": r["key"]}, ExpiresIn=7 * 86400)
        lists.setdefault(k, []).append((url, r["key"], r["size"]))
    os.makedirs(OUT, exist_ok=True)
    for k, rows in sorted(lists.items()):
        with open(os.path.join(OUT, f"{k}.urls.txt"), "w", encoding="utf-8") as f:
            for row in rows: f.write("\t".join(row) + "\n")
        print(f"{k:20s} {len(rows):5d} files {sum(int(r[2]) for r in rows)/1e9:7.2f} GB")


if __name__ == "__main__":
    main()
