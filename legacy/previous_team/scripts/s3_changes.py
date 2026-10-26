"""Detect training-data changes in the KOZ-3 bucket: new, deleted and modified objects since the last snapshot.
Compares size, ETag and LastModified (organisers replace files in place with the same size, so size alone misses
changes). Writes docs/s3_snapshots/<UTC timestamp>.tsv and prints the diff against the newest previous snapshot.
    python scripts/s3_changes.py"""
import boto3, glob, os, datetime
from botocore.config import Config

s3 = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru",
    aws_access_key_id=os.environ["KOZ3_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["KOZ3_S3_SECRET_KEY"],
    region_name="ru-7", config=Config(s3={"addressing_style": "path"}))
SNAP = "docs/s3_snapshots"; os.makedirs(SNAP, exist_ok=True)
cur = {}
for pg in s3.get_paginator("list_objects_v2").paginate(Bucket="train-expds-3"):
    for o in pg.get("Contents", []):
        cur[o["Key"]] = (o["Size"], o["ETag"].strip('"'), o["LastModified"].strftime("%Y-%m-%d %H:%M"))
prev_files = sorted(glob.glob(os.path.join(SNAP, "*.tsv")))
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M")
with open(os.path.join(SNAP, f"{stamp}.tsv"), "w", encoding="utf-8") as f:
    for k, (s, e, m) in sorted(cur.items()):
        f.write(f"{s}\t{e}\t{m}\t{k}\n")
print(f"{len(cur)} objects, {sum(v[0] for v in cur.values()) / 1e9:.2f} GB, {len({k.split('/')[0] for k in cur})} regions")
if not prev_files:
    print("first snapshot written; nothing to compare"); raise SystemExit
prev = {}
for l in open(prev_files[-1], encoding="utf-8"):
    s, e, m, k = l.rstrip("\n").split("\t", 3); prev[k] = (int(s), e, m)
print("compared with", os.path.basename(prev_files[-1]), "(times UTC; Moscow = UTC+3)")
for k in sorted(set(cur) - set(prev)):
    print(f"  NEW      {cur[k][2]}  {cur[k][0]:>12}  {k}")
for k in sorted(set(prev) - set(cur)):
    print(f"  DELETED                     {k}")
for k in sorted(set(cur) & set(prev)):
    if cur[k][:2] != prev[k][:2]:
        print(f"  MODIFIED {cur[k][2]}  {prev[k][0]:>12} -> {cur[k][0]:<12} {k}")
