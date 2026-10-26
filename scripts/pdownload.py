"""Parallel resumable downloader for the signed-URL lists. Skips files already at the right size.

Usage: python scripts/pdownload.py <list.urls.txt> [dest=data] [workers=16]
"""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor


def fetch(url, out, size):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for _ in range(30):
        if os.path.exists(out) and os.path.getsize(out) == size:
            return True
        subprocess.run(["curl", "-L", "-s", "-S", "-C", "-", "--retry", "8", "--retry-delay", "5",
                        "--speed-limit", "10000", "--speed-time", "60", "--connect-timeout", "20",
                        "--max-time", str(max(120, size // 200_000)), "-o", out, url], check=False)
    return os.path.exists(out) and os.path.getsize(out) == size


def main():
    lst, dest = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "data")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    jobs = []
    for line in open(lst, encoding="utf-8"):
        url, key, size = line.rstrip("\n").split("\t")
        jobs.append((url, os.path.join(dest, "train-expds-3", key), int(size)))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: fetch(*j), jobs))
    bad = [j[1] for j, ok in zip(jobs, res) if not ok]
    print(f"{os.path.basename(lst)}: {len(jobs) - len(bad)}/{len(jobs)} complete", flush=True)
    for b in bad:
        print("  INCOMPLETE", b, flush=True)


if __name__ == "__main__":
    main()
