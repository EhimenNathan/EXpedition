"""Assemble the submission folder and archive from inference/ (Technical Regulation: archive <= 2 GB with
metadata.json + inference/main.py; no training or validation data).

    python scripts/build_submission.py [--out build/submission] [--image <docker image>]

Copies only what the run needs: inference/{main.py, run.py, koz3/*.py}, models/config.json, the model files that
config.json references (lidar, SpOr, SpOr_det, Ae, Ae_det, Or), the detector config folders, and models/geo/.
Writes <out>/submission/ and <out>/submission.zip, and prints the size against the 2 GB limit."""
import argparse, glob, json, os, shutil, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "build"))
ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")  # placeholder until the Docker step
a = ap.parse_args()
SRC = os.path.join(ROOT, "inference"); DST = os.path.join(a.out, "submission"); INF = os.path.join(DST, "inference")
shutil.rmtree(DST, ignore_errors=True); os.makedirs(os.path.join(INF, "koz3")); os.makedirs(os.path.join(INF, "models"))
for f in ("main.py", "run.py"):
    shutil.copy2(os.path.join(SRC, f), INF)
for f in glob.glob(os.path.join(SRC, "koz3", "*.py")):
    shutil.copy2(f, os.path.join(INF, "koz3"))
cfg = json.load(open(os.path.join(SRC, "models", "config.json"), encoding="utf-8"))
shutil.copy2(os.path.join(SRC, "models", "config.json"), os.path.join(INF, "models"))
need = set()
for k, v in cfg.items():
    if isinstance(v, list):
        need |= {s["file"] for s in v if isinstance(s, dict) and "file" in s}
    elif isinstance(v, dict) and "files" in v:
        need |= set(v["files"]); need.add(v["config_dir"])
for n in sorted(need) + ["geo"]:
    s = os.path.join(SRC, "models", n)
    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, os.path.join(INF, "models", n))
json.dump({"image": a.image, "entry_point": "python inference/main.py /data/input /data/output/result.geojson"},
          open(os.path.join(DST, "metadata.json"), "w"), indent=1)
zp = os.path.join(a.out, "submission.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:  # weights do not compress; stored keeps it fast
    for p in glob.glob(os.path.join(DST, "**", "*"), recursive=True):
        if os.path.isfile(p) and "__pycache__" not in p:
            z.write(p, os.path.relpath(p, DST))
sz = os.path.getsize(zp)
print(f"models: {sorted(need)}\n{zp}: {sz / 1e9:.3f} GB ({'OK' if sz < 2e9 else 'OVER'} vs 2 GB)")
