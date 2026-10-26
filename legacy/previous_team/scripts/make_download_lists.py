"""Generate signed (pre-signed, 7-day) download links for the KOZ-3 bucket, grouped by data type.

    python scripts/s3_changes.py              # 1) refresh the bucket snapshot (also reports silent changes)
    python scripts/make_download_lists.py     # 2) write docs/download_lists/<group>.urls.txt (+ .aria2.txt)

Each .urls.txt line: <signed URL> TAB <bucket key> TAB <size in bytes>. Download with
    Windows:  powershell -ExecutionPolicy Bypass -File docs/download_lists/download.ps1 -List <file> -Dest incoming
    Linux:    bash docs/download_lists/download.sh <file> incoming
then move verified files into place with  python scripts/ingest_incoming.py
Why signed links + curl: on the original PC boto3 transfers ran at ~20 KB/s and timed out, curl at ~5.6 MB/s.
Links expire after 7 days; just rerun this script."""
import glob, os, re
import boto3
from botocore.config import Config

s3 = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru",
                  aws_access_key_id=os.environ["KOZ3_S3_ACCESS_KEY"],
                  aws_secret_access_key=os.environ["KOZ3_S3_SECRET_KEY"],
                  region_name="ru-7", config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"))
snap = sorted(glob.glob("docs/s3_snapshots/*.tsv"))[-1]
rows = []
for l in open(snap, encoding="utf-8"):
    s, e, m, k = l.rstrip("\n").split("\t", 3)
    if not k.endswith("/"):
        rows.append((int(s), k))
size = {k: s for s, k in rows}
geo_re = re.compile(r"^2(0\d|1\d)_")
sites_with = lambda pat: {k.split("/")[0] for _, k in rows if re.search(pat, k)}
sp_sites, ae_sites, or_sites = sites_with(r"_SpOr_"), sites_with(r"_(Ae|Ае)_"), sites_with(r"_Or_")
utm = lambda k, sites: k.endswith("/UTM.json") and k.split("/")[0] in sites
GROUPS = {
    "1_geophysics": [k for _, k in rows if geo_re.match(k) and not k.endswith(".las") and "_Li_" not in k],
    "2_satellite_SpOr": [k for _, k in rows if not geo_re.match(k) and ("_SpOr_" in k or utm(k, sp_sites))],
    "3_aerial_Ae": [k for _, k in rows if re.search(r"_(Ae|Ае)_", k) or utm(k, ae_sites)],
    "4_lidar_LAS": sorted([k for _, k in rows if k.endswith(".las")] + [k for _, k in rows if re.search(r"_Li_.*\.geojson$", k)]
                          + [k for _, k in rows if k.endswith("/UTM.json") and any(x.startswith(k.split("/")[0] + "/") and x.endswith(".las") for _, x in rows)],
                          key=lambda k: size[k]),
    "5_orthophoto_Or": [k for _, k in rows if "_Or_" in k or utm(k, or_sites)],
    "6_lidar_maps_Li": [k for _, k in rows if "_Li_" in k and k.endswith(".tif")],
}
os.makedirs("docs/download_lists", exist_ok=True)
for name, keys in GROUPS.items():
    keys = list(dict.fromkeys(keys))
    with open(f"docs/download_lists/{name}.urls.txt", "w", encoding="utf-8") as fu, open(f"docs/download_lists/{name}.aria2.txt", "w", encoding="utf-8") as fa:
        for k in keys:
            u = s3.generate_presigned_url("get_object", Params={"Bucket": "train-expds-3", "Key": k}, ExpiresIn=7 * 24 * 3600)
            fu.write(f"{u}\t{k}\t{size[k]}\n"); fa.write(f"{u}\n  out=train-expds-3/{k}\n")
    print(f"{name}: {len(keys)} files, {sum(size[k] for k in keys) / 1e9:.2f} GB")
