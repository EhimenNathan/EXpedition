"""Download a subset of the KOZ-3 bucket, resumably (skips files already present with the right size).
    python scripts/s3_fetch.py <dest_dir> <key_regex> [--exclude <regex>] [--max-mb N]
Example (all geophysics except LAS and LiDAR maps):
    python scripts/s3_fetch.py data_geo "^2(0\\d|1\\d)_" --exclude "\\.las$|_Li_"
Reads the newest snapshot in docs/s3_snapshots/ (run scripts/s3_changes.py first) and prints throughput."""
import argparse, glob, os, re, time
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

ap = argparse.ArgumentParser()
ap.add_argument("dest"); ap.add_argument("pattern"); ap.add_argument("--exclude", default=None)
ap.add_argument("--max-mb", type=float, default=None)
a = ap.parse_args()
s3 = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru",
    aws_access_key_id=os.environ["KOZ3_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["KOZ3_S3_SECRET_KEY"],
    region_name="ru-7", config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10, "mode": "adaptive"}, connect_timeout=30, read_timeout=120))
tcfg = TransferConfig(multipart_threshold=64 * 1024 * 1024, max_concurrency=3, multipart_chunksize=16 * 1024 * 1024)
snap = sorted(glob.glob("docs/s3_snapshots/*.tsv"))[-1]
todo = []
for l in open(snap, encoding="utf-8"):
    s, e, m, k = l.rstrip("\n").split("\t", 3); s = int(s)
    if k.endswith("/") or not re.search(a.pattern, k) or (a.exclude and re.search(a.exclude, k)):
        continue
    if a.max_mb and s > a.max_mb * 1e6:
        continue
    out = os.path.join(a.dest, k)
    if os.path.exists(out) and os.path.getsize(out) == s:
        continue
    todo.append((k, s, out))
todo.sort(key=lambda t: t[1])  # small files first: analysis can start while the big ones arrive
tot = sum(s for _, s, _ in todo); print(f"{len(todo)} files, {tot / 1e6:.0f} MB to fetch", flush=True)
t0 = time.time(); done = 0
import threading
from concurrent.futures import ThreadPoolExecutor
lock = threading.Lock()


def fetch(item):
    global done
    k, s, out = item
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(5):  # read_timeout turns a silently dead connection into an exception; retry the file
        try:
            s3.download_file("train-expds-3", k, out, Config=tcfg); break
        except Exception as e:
            print(f"retry {attempt + 1} {k}: {e!r}", flush=True); time.sleep(10)
    with lock:
        done += s; dt = time.time() - t0
        print(f"{done / 1e6:8.0f}/{tot / 1e6:.0f} MB  {done / 1e6 / max(dt, 1e-3):5.1f} MB/s  {k}", flush=True)


small = [t for t in todo if t[1] < 8e6]; big = [t for t in todo if t[1] >= 8e6]
with ThreadPoolExecutor(4) as ex:  # small files are latency-bound: fetch many at once
    list(ex.map(fetch, small))
for t in big:  # big files: one at a time, each already multipart-parallel
    fetch(t)
print("done", flush=True)
