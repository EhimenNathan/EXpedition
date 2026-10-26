"""Platform rehearsal: unzip the archive, run `python inference/main.py IN OUT/result.geojson` like the platform,
watch the output dir every 100 ms (first appearance, every (size, mtime) seen), then score against GT.

  python scripts/rehearse.py build/koz3_submission.zip work/rehearsal/input work/rehearsal/gt.geojson
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def main():
    zp, inp, gt = sys.argv[1:4]
    work = tempfile.mkdtemp(prefix="koz3_rehearse_", dir=os.path.join(ROOT, "work"))
    zipfile.ZipFile(zp).extractall(os.path.join(work, "app"))
    out = os.path.join(work, "output"); os.makedirs(out)
    seen, files, stop = [], set(), threading.Event()
    t0 = time.time()

    def watch():
        last = None
        while not stop.is_set():
            for f in os.listdir(out):
                files.add(f)
            p = os.path.join(out, "result.geojson")
            if os.path.exists(p):
                st = os.stat(p); cur = (st.st_size, st.st_mtime)
                if cur != last:
                    seen.append((round(time.time() - t0, 2), cur)); last = cur
            time.sleep(0.1)
    th = threading.Thread(target=watch); th.start()
    py = os.environ.get("REHEARSE_PY", sys.executable)   # e.g. work/plat312/bin/python: the platform-like env
    env = dict(os.environ)
    if os.environ.get("REHEARSE_BLOCK") == "1":      # block packages the platform image lacks (scripts/platform_sim)
        env["PYTHONPATH"] = os.path.join(ROOT, "scripts", "platform_sim")
    r = subprocess.run([py, "inference/main.py", os.path.abspath(inp), os.path.join(out, "result.geojson")],
                       cwd=os.path.join(work, "app"), env=env)
    wall = time.time() - t0
    time.sleep(0.5); stop.set(); th.join()
    print(f"exit {r.returncode} wall {wall:.1f}s; files seen {sorted(files)}; result.geojson states {seen}")
    print("PASS (single atomic write)" if len(seen) == 1 else "FAIL: result.geojson changed after appearing")
    res = json.load(open(os.path.join(out, "result.geojson")))
    from collections import Counter
    print(len(res["features"]), "features", Counter((f["properties"]["class_name"], f["properties"]["markup_type"]) for f in res["features"]))
    import compute_metrics as cm
    off = cm.F1Calculator().calculate_f1(res, json.load(open(gt)))
    print(f"OFFICIAL final_score {off['final_score']:.4f}")
    for c, v in off["per_class"].items():
        if v["num_ground_truth"] or v["num_predictions"]:
            print(f"  {c:24s} f1 {v['f1']:.3f} tp {v['tp']} fp {v['fp']} fn {v['fn']}")
    shutil.copy(os.path.join(out, "result.geojson"), os.path.join(os.path.dirname(gt), "result.geojson"))
    if os.environ.get("KEEP_REHEARSAL") != "1":       # the unpacked archive is ~1 GB: do not let rehearsals fill the disk
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
