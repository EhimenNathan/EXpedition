import os
os.environ['KOZ3_DATA_NAME']='sp03'
os.environ['KOZ3_TAG']='sp03_dfine_l'
# ---- kcommon: shared preamble of every KOZ-3 kernel on the sola account (prepended by kaggle_io.push) ----
# Data access uses the 7-day pre-signed links in docs/download_lists/*.urls.txt (no credentials in kernels).
import glob as _glob, os as _os, sys as _sys, tarfile as _tarfile, subprocess as _sp
CODE = _os.environ.get("KOZ3_CODE", "/kaggle/code_root")  # overridable for local smoke tests
INPUT = _os.environ.get("KOZ3_INPUT", "/kaggle/input")
if not _os.path.exists(f"{CODE}/koz3"):
    _t = _glob.glob(f"{INPUT}/**/koz3_code.tar.gz", recursive=True)
    if _t:
        _os.makedirs(CODE, exist_ok=True); _tarfile.open(_t[0]).extractall(CODE)
    else:  # Kaggle auto-extracted the archive
        _src = _os.path.dirname(_os.path.dirname(_glob.glob(f"{INPUT}/**/koz3/sp_io.py", recursive=True)[0]))
        import shutil as _sh; _sh.copytree(_src, CODE, dirs_exist_ok=True)  # copy: /kaggle/input is read-only
_sys.path.insert(0, CODE); _os.chdir(CODE)


def need(*mods):
    """pip-install modules missing from the image (internet is on). "module:pip-name" when the names differ."""
    import importlib.util
    miss = [m.split(":")[-1] for m in mods if importlib.util.find_spec(m.split(":")[0]) is None]
    if miss:
        _sp.run([_sys.executable, "-m", "pip", "install", "-q", *miss], check=True)


def url_list(name):
    """(url, key, size) rows of docs/download_lists/<name>.urls.txt"""
    rows = []
    for line in open(f"{CODE}/docs/download_lists/{name}.urls.txt", encoding="utf-8"):
        if line.strip():
            u, k, s = line.rstrip("\n").split("\t"); rows.append((u, k, int(s)))
    return rows


def fetch(rows, dest_root, threads=8):
    """Download rows to dest_root/<key> in parallel with retries; skip complete files. Returns bytes fetched."""
    import requests
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        u, k, sz = r; p = _os.path.join(dest_root, k)
        if k.endswith("/") or (_os.path.exists(p) and _os.path.getsize(p) == sz):
            return 0
        _os.makedirs(_os.path.dirname(p), exist_ok=True)
        for attempt in range(10):
            try:
                with requests.get(u, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(p, "wb") as f:
                        for b in resp.iter_content(1 << 22):
                            f.write(b)
                if _os.path.getsize(p) == sz:
                    return sz
            except Exception as e:
                print("retry", k, attempt, repr(e)[:120], flush=True)
                import random, time
                time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))  # S3 answers 429 under load: back off
        raise RuntimeError(f"download failed: {k}")
    with ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(one, rows))


def run(cmd, env=None, log=None):
    print("$", " ".join(cmd), flush=True)
    if log:
        return _sp.Popen(cmd, env={**_os.environ, **(env or {})}, stdout=open(log, "w"), stderr=_sp.STDOUT)
    return _sp.run(cmd, env={**_os.environ, **(env or {})}).returncode
# ---- end kcommon ----


"""KOZ-3 satellite detector family on T4x2: scripts/train_sp_det.py (D-FINE, HF transformers), two lanes (one per GPU),
4 grouped folds identical to the U-Net's -> full out-of-fold set in the scripts/eval_sp.py schema.
Input: the koz3-sp-prep kernel output (kernel source). Env: KOZ3_DATA_NAME (sp|ae), KOZ3_TAG, KOZ3_ARGS, KOZ3_SPLIT ("0,1;2,3"), KOZ3_WORKERS."""
import glob, os, subprocess, sys, time
t0 = time.time()
need("rasterio", "pyarrow", "shapely")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51"], check=True)
DATA_NAME = os.environ.get("KOZ3_DATA_NAME", "sp")  # prepared folder in the input kernel's output: sp (satellite) or ae
_hits = glob.glob(f"{INPUT}/**/{DATA_NAME}/scenes.csv", recursive=True) or glob.glob(f"{INPUT}/**/scenes.csv", recursive=True)
print("scenes.csv candidates:", _hits, flush=True)
SP = os.environ.get("KOZ3_SP") or os.path.dirname(_hits[0])
TAG = os.environ.get("KOZ3_TAG", "sp_dfine_l")
ARGS = os.environ.get("KOZ3_ARGS", "--ckpt ustc-community/dfine-large-coco --chip 640 --bs 8 --iters 8000 --lr 1e-4").split()
OUTROOT = os.environ.get("KOZ3_OUTROOT", "/kaggle/working/sp_runs")
try:
    import torch; ngpu = torch.cuda.device_count()
except Exception:
    ngpu = 0
split = os.environ.get("KOZ3_SPLIT", "0,1;2,3" if ngpu >= 2 else "0,1,2,3").split(";")
cpus = os.cpu_count() or 2; workers = int(os.environ.get("KOZ3_WORKERS", max(1, cpus // len(split) - 1)))
print("data", SP, "gpus", ngpu, "cpus", cpus, "lanes", split, "workers/lane", workers, flush=True)
os.makedirs(f"{OUTROOT}/{TAG}", exist_ok=True)
assert run([sys.executable, "scripts/train_sp_det.py", "--data", SP, "--folds", "0", "--tag", "_preflight", "--smoke",
            "--outroot", os.path.join(os.environ.get("KOZ3_TMP", "/tmp"), "det_pre"), "--workers", str(workers),
            *[x for x in ARGS if x != "--smoke"]], env={"PYTHONIOENCODING": "utf-8"}) == 0
procs = []
for i, folds in enumerate(split):
    env = {"PYTHONIOENCODING": "utf-8"}
    if ngpu:
        env["CUDA_VISIBLE_DEVICES"] = str(i % ngpu)
    procs.append(run([sys.executable, "scripts/train_sp_det.py", "--data", SP, "--folds", folds, "--tag", TAG,
                      "--outroot", OUTROOT, "--workers", str(workers), *ARGS], env=env, log=f"{OUTROOT}/{TAG}/lane{i}.log"))
while any(p.poll() is None for p in procs):
    time.sleep(120)
    for i in range(len(procs)):
        tail = open(f"{OUTROOT}/{TAG}/lane{i}.log", encoding="utf-8", errors="replace").read().strip().splitlines()[-1:]
        print(f"[{time.time() - t0:.0f}s] lane{i}:", tail[0][:160] if tail else "", flush=True)
rcs = [p.returncode for p in procs]
print("rcs", rcs, "DONE", round(time.time() - t0), "s", flush=True)
assert all(rc == 0 for rc in rcs)
