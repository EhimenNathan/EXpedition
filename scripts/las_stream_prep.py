"""Stream every training LAS: download → DTM (koz3.dtm) → data/las_prep/<region>.npz → delete the raw LAS.

Raw clouds are 40 GB and the disk cannot hold them next to everything else; the DTMs are a few GB.
Re-running skips regions already prepared. Usage: python scripts/las_stream_prep.py [--keep-raw] [workers=3]
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3 import dtm  # noqa: E402
from koz3.labels import DATA, region_utm  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LIST = os.path.join(ROOT, "docs", "download_lists", "4_lidar_LAS.urls.txt")
OUT = os.path.join(ROOT, "data", "las_prep")


def one(job, keep_raw):
    url, key, size = job
    region = key.split("/")[0]
    out = os.path.join(OUT, region + ".npz")
    if os.path.exists(out):
        return
    las = os.path.join(DATA, key)
    os.makedirs(os.path.dirname(las), exist_ok=True)
    for _ in range(40):
        if os.path.exists(las) and os.path.getsize(las) == int(size):
            break
        subprocess.run(["curl", "-L", "-s", "-S", "-C", "-", "--retry", "8", "--connect-timeout", "20",
                        "--speed-limit", "10000", "--speed-time", "60", "-o", las, url], check=False)
    if not (os.path.exists(las) and os.path.getsize(las) == int(size)):
        print("DOWNLOAD FAILED", key, flush=True); return
    t = time.time()
    try:
        d, v, origin = dtm.build(las)
        dtm.save(out, d, v, origin, region_utm(os.path.join(DATA, region)))
        print(f"{region:36s} {int(size)/1e6:8.1f} MB → {d.shape} res {origin[2]} valid {v.mean():.2f} {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print("DTM FAILED", key, repr(e), flush=True); return
    if not keep_raw:
        os.remove(las)


def main():
    keep = "--keep-raw" in sys.argv
    nw = int(next((a for a in sys.argv[1:] if a.isdigit()), 3))
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted((l.rstrip("\n").split("\t") for l in open(LIST, encoding="utf-8")), key=lambda j: int(j[2]))
    with ThreadPoolExecutor(nw) as ex:
        list(ex.map(lambda j: one(j, keep), jobs))
    print("LAS_DONE", flush=True)


if __name__ == "__main__":
    main()
