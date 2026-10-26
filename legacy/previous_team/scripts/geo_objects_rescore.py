"""Re-score geophysics objects from saved out-of-fold probability maps (no retraining), e.g. after a metric update.

    python scripts/geo_objects_rescore.py [--oof work/geo/objects_v3/oof_orig] [--tag _v150]

Uses the current koz3/official/fastmatch (metric koz3-1.5.0: representative point inside the label) and the same
post-processing grid as geo_objects_v3; post-processing is chosen per method on the OTHER regions (honest).
Writes work/geo/objects_v3/counts<tag>.csv and report<tag>.csv."""
import argparse, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import geo_data as GD
from koz3.geo_objects_core import GSD, polygons
from koz3.official.fastmatch import soft_match, prf

ap = argparse.ArgumentParser(); ap.add_argument("--oof", default="work/geo/objects_v3/oof_orig"); ap.add_argument("--tag", default="_v150")
a = ap.parse_args()
U = {f"{u['region']}|{u['method']}|{u['key']}": u for u in GD.discover("data_geo") if u["objects"]}
GRID = [(s, t, m) for s in (0.0, 1.0, 2.0) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for m in (0.3, 1.0, 3.0)]
rows = []
for f in sorted(glob.glob(f"{a.oof}/*.npz")):
    uid = os.path.basename(f)[:-4].replace("__", "|")
    u = U.get(uid)
    if u is None:
        continue
    z = np.load(f); prob = z["prob"].astype(np.float32); v = z["valid"]
    d = GD.load_unit(u, GSD[u["method"]])
    for s_, t_, m_ in GRID:
        tp, fp, fn, _ = soft_match(polygons(prob, v, d["T"], d["gsd"], s_, t_, m_), d["objects"])
        rows.append(dict(uid=uid, region=u["region"], method=u["method"], sigma=s_, thr=t_, min_area=m_, tp=tp, fp=fp, fn=fn))
R = pd.DataFrame(rows); R.to_csv(f"work/geo/objects_v3/counts{a.tag}.csv", index=False)
key = ["sigma", "thr", "min_area"]; out = []
for meth, g in R.groupby("method"):
    tot = np.zeros(3)
    for r in g.region.unique():
        o = g[g.region != r].groupby(key)[["tp", "fp", "fn"]].sum()
        if not len(o):
            continue
        b = max(o.index, key=lambda k: prf(*o.loc[k].values)[2])
        tot += g[(g.region == r) & (g[key].apply(tuple, axis=1) == b)][["tp", "fp", "fn"]].sum().values
    allg = g.groupby(key)[["tp", "fp", "fn"]].sum(); ob = max(allg.index, key=lambda k: prf(*allg.loc[k].values)[2])
    out.append(dict(method=meth, tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), F1_honest=round(prf(*tot)[2], 4), deploy=ob,
                    F1_oracle=round(prf(*allg.loc[ob].values)[2], 4)))
allc = np.sum([[x["tp"], x["fp"], x["fn"]] for x in out], axis=0)
out.append(dict(method="ALL (pooled)", tp=int(allc[0]), fp=int(allc[1]), fn=int(allc[2]), F1_honest=round(prf(*allc)[2], 4)))
O = pd.DataFrame(out); O.to_csv(f"work/geo/objects_v3/report{a.tag}.csv", index=False); pd.set_option("display.width", 200); print(O.to_string(index=False))
