"""Snapshot the train-expds-3 bucket and report NEW / MODIFIED / DELETED files vs the last snapshot.

Compares size + ETag + LastModified (organisers replace files in place, often with the same size).
Usage: python scripts/s3_changes.py
"""
import csv, datetime as dt, glob, os, sys
import boto3
from botocore.config import Config

ENDPOINT = "https://s3.ru-7.storage.selcloud.ru"
BUCKET = "train-expds-3"
# Set these to the public read-only credentials from the competition page.
# They are intentionally not committed, even though the organisers publish
# them, so secret scanners and downstream forks do not mistake them for private
# credentials.
AK = os.environ["KOZ3_S3_ACCESS_KEY"]
SK = os.environ["KOZ3_S3_SECRET_KEY"]
SNAP_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "s3_snapshots")


def client():
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=AK, aws_secret_access_key=SK,
                        region_name="ru-7", config=Config(signature_version="s3v4", retries={"max_attempts": 8}))


def listing():
    rows = {}
    for page in client().get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        for o in page.get("Contents", []):
            rows[o["Key"]] = (o["Size"], o["ETag"].strip('"'), o["LastModified"].isoformat())
    return rows


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["key"]: (int(r["size"]), r["etag"], r["mtime"]) for r in csv.DictReader(f)}


def main():
    os.makedirs(SNAP_DIR, exist_ok=True)
    cur = listing()
    prev_files = sorted(glob.glob(os.path.join(SNAP_DIR, "snapshot_*.csv")))
    prev = load(prev_files[-1]) if prev_files else {}
    total = sum(v[0] for v in cur.values())
    print(f"{len(cur)} objects, {total/1e9:.1f} GB, {len({k.split('/')[0] for k in cur})} top-level folders")
    if prev:
        new = sorted(set(cur) - set(prev)); gone = sorted(set(prev) - set(cur))
        mod = sorted(k for k in set(cur) & set(prev) if cur[k] != prev[k])
        print(f"vs {os.path.basename(prev_files[-1])}: NEW {len(new)}  MODIFIED {len(mod)}  DELETED {len(gone)}")
        for tag, ks in (("NEW", new), ("MODIFIED", mod), ("DELETED", gone)):
            for k in ks:
                print(f"  {tag:8s} {k}  {cur.get(k, prev.get(k))[0]/1e6:.1f} MB")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(SNAP_DIR, f"snapshot_{stamp}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key", "size", "etag", "mtime"])
        for k in sorted(cur):
            w.writerow([k, *cur[k]])
    print("saved", out)


if __name__ == "__main__":
    sys.exit(main())
