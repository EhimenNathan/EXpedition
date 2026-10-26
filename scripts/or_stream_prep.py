"""Stream the 65 GB of orthophotos: download one → chips at 0.1 m (koz3.sp_prep --mod Or --append) → delete raw."""
import os, subprocess, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LIST = os.path.join(ROOT, "docs", "download_lists", "5_orthophoto_Or.urls.txt")
DATA = os.path.join(ROOT, "data", "train-expds-3")
jobs = sorted((l.rstrip("\n").split("\t") for l in open(LIST, encoding="utf-8")), key=lambda j: int(j[2]))
done_f = os.path.join(ROOT, "data", "or_prep_done.txt")
done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
for url, key, size in jobs:
    if key in done:
        continue
    out = os.path.join(DATA, key); os.makedirs(os.path.dirname(out), exist_ok=True)
    for _ in range(60):
        if os.path.exists(out) and os.path.getsize(out) == int(size): break
        subprocess.run(["curl", "-L", "-s", "-S", "-C", "-", "--retry", "8", "--connect-timeout", "20",
                        "--speed-limit", "10000", "--speed-time", "60", "-o", out, url])
    if os.path.getsize(out) != int(size):
        print("DOWNLOAD FAILED", key, flush=True); continue
    r = subprocess.run([sys.executable, "-m", "koz3.sp_prep", "--mod", "Or", "--res", "0.1", "--append"], cwd=ROOT)
    if r.returncode == 0:
        os.remove(out); open(done_f, "a").write(key + "\n"); print("OR done", key, flush=True)
print("OR_DONE", flush=True)
