import os
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
    """pip-install modules missing from the image (internet is on)."""
    import importlib.util
    miss = [m for m in mods if importlib.util.find_spec(m) is None]
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

"""Private CPU-only KOZ-3 OOF check; uses existing private kernel outputs only."""
import glob, json
import numpy as np
import pandas as pd
from shapely import wkb
from koz3.official.fastmatch import soft_match, prf
from koz3.official.compute_metrics import CLASS_WEIGHTS, normalize_class_name

runs = {}
for tag in ('koz3-sp-det','koz3-sp03-det'):
    hits = sorted(glob.glob(f'/kaggle/input/{tag}/**/oof_pred01.parquet',recursive=True))
    if not hits:
        hits = sorted(x for x in glob.glob('/kaggle/input/**/oof_pred01.parquet',recursive=True) if tag in x)
    assert len(hits) == 1,(tag,hits)
    d = os.path.dirname(hits[0]); print(tag,'source',d,flush=True)
    pred = pd.concat([pd.read_parquet(f'{d}/oof_pred01.parquet'),pd.read_parquet(f'{d}/oof_pred23.parquet')],ignore_index=True)
    gt = pd.read_parquet(f'{d}/oof_gt.parquet')
    pred['class_name']=pred.cls.map(normalize_class_name);gt['class_name']=gt.cls.map(normalize_class_name)
    pred['geom']=[wkb.loads(x) for x in pred.wkb];gt['geom']=[wkb.loads(x) for x in gt.wkb]
    runs[tag]=(pred,gt)

common=set.intersection(*(set(p.site) for p,_ in runs.values()))
print('common sites',len(common),flush=True)
thresholds=(.25,.35,.45,.55,.65)
results={}
for tag,(pred,gt) in runs.items():
    pred=pred[pred.site.isin(common)];gt=gt[gt.site.isin(common)]
    records=[]
    for cls in CLASS_WEIGHTS:
        gl=gt[gt.class_name==cls]
        if gl.empty: continue
        for threshold in thresholds:
            pl=pred[(pred.class_name==cls)&(pred.score>=threshold)]
            tp=fp=fn=0
            for site in common:
                pp=pl[pl.site==site];gg=gl[gl.site==site]
                if pp.empty and gg.empty: continue
                a,b,c,_=soft_match(list(zip(pp.geom,pp.score)),list(gg.geom))
                tp+=a;fp+=b;fn+=c
            records.append(dict(cls=cls,threshold=threshold,tp=tp,fp=fp,fn=fn,f1=prf(tp,fp,fn)[2]))
    table=pd.DataFrame(records)
    scores={str(t):float(sum(CLASS_WEIGHTS[c]*table[(table.cls==c)&(table.threshold==t)].f1.iloc[0] for c in table.cls.unique())/sum(CLASS_WEIGHTS[c] for c in table.cls.unique())) for t in thresholds}
    results[tag]={'n_pred':int(len(pred)),'n_gt':int(len(gt)),'scores':scores,'best_fixed':max(scores,key=scores.get)}
    print(tag,results[tag],flush=True)
with open('/kaggle/working/codex_oof_check.json','w') as f: json.dump(results,f,indent=2)
