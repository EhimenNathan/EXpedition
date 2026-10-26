"""Run the organisers' baseline (docs/org_20261014/baseline/solution.py, YOLO11s-seg) region by region, resumably.

EVALUATION ONLY. The baseline needs ultralytics (AGPL-3.0); it lives in the separate venv tools/yolo_eval_venv and is
never imported by our solution (inference/ must not depend on it).

    python scripts/baseline_eval.py run   --src data_sp --tag sp                 # every region folder under data_sp
    python scripts/baseline_eval.py run   --src work/baseline_eval/las_in --tag las
    python scripts/baseline_eval.py fetch --sites 030_Эсбю,036_Тимерево ...       # LAS for LiDAR sites (signed links)

Each region runs as its own baseline process with its CLI defaults (conf 0.25, IoU 0.5, tiles 1024 / 30 % overlap,
CLAHE on, hillshades c/ch/g/i at 0.1 m from LAS, extra rasters or+sp_or), so a crash loses one region only.
The input of one run is work/baseline_eval/<tag>/_in/<region>/<region>, a directory junction to the real folder.
Outputs: work/baseline_eval/<tag>/<region>/result.geojson + log.txt + seconds.txt. Score with scripts/baseline_score.py."""
import argparse, json, os, re, shutil, subprocess, sys, time, _winapi

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PY = os.path.join(ROOT, "tools", "yolo_eval_venv", "Scripts", "python.exe")
SOL = os.path.join(ROOT, "docs", "org_20261014", "baseline", "solution.py")
MODEL = os.path.join(ROOT, "docs", "org_20261014", "baseline", "model.pt")


def run(a):
    out_root = os.path.join(ROOT, "work", "baseline_eval", a.tag)
    regions = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)) and not d.startswith("_"))
    if a.sites:
        keep = set(a.sites.split(","))
        regions = [r for r in regions if r in keep or r[:3] in keep]
    env = dict(os.environ, YOLO_OFFLINE="1", PYTHONIOENCODING="utf-8")
    for i, reg in enumerate(regions, 1):
        od = os.path.join(out_root, reg)
        if os.path.exists(os.path.join(od, "result.geojson")):
            continue
        os.makedirs(od, exist_ok=True)
        holder = os.path.join(out_root, "_in", reg); os.makedirs(holder, exist_ok=True)
        link = os.path.join(holder, reg)
        if not os.path.exists(link):
            _winapi.CreateJunction(os.path.abspath(os.path.join(a.src, reg)), link)
        t = time.time()
        with open(os.path.join(od, "log.txt"), "w", encoding="utf-8") as log:
            rc = subprocess.call([PY, SOL, holder, od, MODEL, "--extra-modalities", a.extra], stdout=log, stderr=subprocess.STDOUT, env=env)
        dt = time.time() - t
        open(os.path.join(od, "seconds.txt"), "w").write(f"{dt:.1f} rc={rc}\n")
        print(f"[{i}/{len(regions)}] {reg}: rc={rc} {dt:.0f}s", flush=True)


def fetch(a):
    """Download LAS of the chosen sites into work/baseline_eval/las_in/<site>/<La folder>/<file>, plus UTM.json."""
    lst = os.path.join(ROOT, "docs", "download_lists", "4_lidar_LAS.urls.txt")
    want = a.sites.split(",")
    rows = [l.rstrip("\n").split("\t") for l in open(lst, encoding="utf-8") if l.strip()]
    for url, key, size in rows:
        site = key.split("/")[0]
        if not any(site == w or site[:3] == w for w in want):
            continue
        dst = os.path.join(ROOT, "work", "baseline_eval", "las_in", *key.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        utm = os.path.join(ROOT, "data_meta", site, "UTM.json")
        if os.path.exists(utm):
            shutil.copyfile(utm, os.path.join(ROOT, "work", "baseline_eval", "las_in", site, "UTM.json"))
        for _ in range(5):
            if os.path.exists(dst) and os.path.getsize(dst) == int(size):
                break
            subprocess.call(["curl", "-s", "-L", "-C", "-", "--retry", "5", "-o", dst, url])
        ok = os.path.exists(dst) and os.path.getsize(dst) == int(size)
        print(f"{key}: {'ok' if ok else 'INCOMPLETE'} {int(size) / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "fetch"])
    ap.add_argument("--src", default="data_sp"); ap.add_argument("--tag", default="sp"); ap.add_argument("--sites", default="")
    ap.add_argument("--extra", default="or,sp_or")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else fetch(a)
