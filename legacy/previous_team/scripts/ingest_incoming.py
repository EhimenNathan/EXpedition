"""Move downloaded bucket files from incoming/train-expds-3/ into the working data folders after verifying their size
against the latest bucket snapshot (docs/s3_snapshots/). Files still being written (size mismatch) are left alone.
    python scripts/ingest_incoming.py
Routing: geophysics regions 200-217 -> data_geo/; satellite (SpOr) -> data_sp/; aerial (Ae) -> data_ae/;
orthophoto (Or) -> data_or/; LAS -> data_las/; anything else (labels, UTM.json) -> the folder of its region's modality
set, i.e. the same region goes to every folder that holds its imagery.
Downloads were made with docs/download_lists/download.ps1 (curl, ~5.6 MB/s here vs ~20 KB/s for boto3)."""
import glob, os, re, shutil

SRC = "incoming/train-expds-3"
snap = sorted(glob.glob("docs/s3_snapshots/*.tsv"))[-1]
size = {}
for l in open(snap, encoding="utf-8"):
    s, e, m, k = l.rstrip("\n").split("\t", 3); size[k] = int(s)


def route(k):
    if re.match(r"^2(0\d|1\d)_", k):
        return ["data_geo"]
    if "_SpOr_" in k:
        return ["data_sp"]
    if re.search(r"_(Ae|Ае)_", k):
        return ["data_ae"]
    if "_Or_" in k:
        return ["data_or"]
    if k.endswith(".las"):
        return ["data_las"]
    # shared files (UTM.json, other labels): every modality folder whose imagery exists for this region in the bucket
    reg = k.split("/")[0]
    return [d for d, pat in (("data_sp", "_SpOr_"), ("data_ae", "_Ae_"), ("data_or", "_Or_"), ("data_las", ".las"))
            if any(x.startswith(reg + "/") and pat in x for x in size)] or ["data_misc"]


moved = bad = 0
for p in glob.glob(os.path.join(SRC, "**", "*"), recursive=True):
    if not os.path.isfile(p):
        continue
    k = os.path.relpath(p, SRC).replace(os.sep, "/")
    if k not in size:
        print("not in bucket snapshot:", k); bad += 1; continue
    if os.path.getsize(p) != size[k]:
        continue  # still downloading
    dests = route(k)
    for i, d in enumerate(dests):
        out = os.path.join(d, k); os.makedirs(os.path.dirname(out), exist_ok=True)
        if i < len(dests) - 1:
            shutil.copy2(p, out)
        else:
            shutil.move(p, out)
    moved += 1
print(f"moved {moved} verified files; {bad} unknown")
