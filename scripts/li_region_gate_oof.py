"""LiDAR site-level kurgan gate, nested + robust: drop all kurgan emissions of a site whose best kurgan confidence
< g (deployed blend + prior, thr 0.15, dual 0.05)."""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")
from koz3.li_train import labels, site_items, decode_site_v2, CLS
from koz3.labels import to_gt_geojson
from koz3.nested import fold_counts, nested, weighted, _add_all
from robust_accept import site_counts, report
import li_compare_oof as lc
L = labels(); items = {it["region"]: it for it in site_items(L)}
P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
sites = sorted(P["ens"]); fold = {r: items[r]["fold"] for r in sites}; folds = sorted(set(fold.values()))
pri = {"R": 200.0, "K": 3, "snb": 0.2}; D = {}
for r in sites:
    it = items[r]; b = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32)
    D[r] = decode_site_v2(b, it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05, comp_thr=0.5, prior=pri)
del P
def site_fc(st, r):
    fs = D[r]
    if st[0] != "base":
        km = max([f["properties"]["confidence"] for f in fs if f["properties"]["class_name"] in K], default=0)
        if km < st[0]:
            fs = [f for f in fs if f["properties"]["class_name"] not in K]
    return {"type": "FeatureCollection", "features": fs}
SS = [("base",)] + [(g,) for g in (0.25, 0.3, 0.35, 0.4, 0.5)]
gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
C = fold_counts(lambda st, f: {"type": "FeatureCollection", "features": [g for r in sites if fold[r] == f for g in site_fc(st, r)["features"]]}, SS, folds, gt, CLS)
for st in SS:
    tot = _add_all(C[st].values()); print(f"{st}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  kt {tuple(int(v) for v in tot[K[1]])}", flush=True)
h, ch, o, os_, _ = nested(C)
print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
if os_ != SS[0]:
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    CS = site_counts(site_fc, (SS[0], os_), sites, gtr, CLS); report(f"LiDAR site gate {os_}", CS[SS[0]], CS[os_])
