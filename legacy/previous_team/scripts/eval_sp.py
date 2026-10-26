"""Evaluate satellite out-of-fold candidates (scripts/train_sp.py) under the official KOZ-3 metric.

    python scripts/eval_sp.py <run_dir> [more run dirs for an ensemble are evaluated by scripts/ens_sp.py]

* Matching: official rule (cov >= 0.5 of the smaller shape AND predicted centroid inside the label), one-to-one,
  greedy by confidence, per scene (scenes never overlap), pooled per class.
* Honest thresholds: for each fold, the (mask, confidence) pair per class is chosen on the other folds and applied to it.
* Also reports the oracle (hindsight) number and the "dual kurgan" variant (each kurgan also emitted with the other
  kurgan class at confidence * 0.8), which hedges the intact/damaged call that dominates LiDAR errors.
Writes <run_dir>/eval_official.csv and <run_dir>/deploy_thresholds.json (chosen on all out-of-fold data)."""
import json, os, sys
import numpy as np, pandas as pd
from shapely import wkb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3.official.fastmatch import soft_match, prf
from koz3.official.compute_metrics import CLASS_WEIGHTS, normalize_class_name

RUN = sys.argv[1]
P = pd.read_parquet(f"{RUN}/oof_pred.parquet"); G = pd.read_parquet(f"{RUN}/oof_gt.parquet")
P["lcls"] = P.cls.map(normalize_class_name); G["lcls"] = G.cls.map(normalize_class_name)
P["geom"] = [wkb.loads(x) for x in P.wkb]; G["geom"] = [wkb.loads(x) for x in G.wkb]
G = G[G.fold.isin(P.fold.unique())]
CONF = [float(x) for x in os.environ.get("KOZ3_CONF", "0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7").split(",")]  # detector scores sit lower: pass a lower grid
K = ["kurgany_povrezhdennye", "kurgany_tselye"]


def grid_counts(PP):
    out = {}
    for cls in CLASS_WEIGHTS:
        Gc = G[G.lcls == cls]
        if not len(Gc):
            continue
        for m in sorted(PP.thr.unique()):
            for c in CONF:
                Pc = PP[(PP.lcls == cls) & (PP.thr == m) & (PP.score >= c)]
                per = {}
                for f in sorted(G.fold.unique()):
                    tp = fp = fn = 0
                    pf, gf = Pc[Pc.fold == f], Gc[Gc.fold == f]
                    for s in set(pf.scene) | set(gf.scene):
                        ps = pf[pf.scene == s]
                        a, b, d, _ = soft_match(list(zip(ps.geom, ps.score)), list(gf[gf.scene == s].geom))
                        tp += a; fp += b; fn += d
                    per[f] = (tp, fp, fn)
                out.setdefault(cls, {})[(m, c)] = per
    return out


def summarise(grid, name):
    rows, folds = [], sorted(G.fold.unique())
    for cls, g in grid.items():
        tot = np.zeros(3); ff = []
        for f in folds:
            oth = [o for o in folds if o != f] or [f]  # single-fold runs (smoke) have no other fold
            best = max(g, key=lambda k: prf(*np.sum([g[k][o] for o in oth], axis=0))[2])
            tot += g[best][f]; ff.append(round(prf(*g[best][f])[2], 3))
        orc = max(g, key=lambda k: prf(*np.sum(list(g[k].values()), axis=0))[2])
        rows.append(dict(variant=name, cls=cls, F1_honest=round(prf(*tot)[2], 4), tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]),
                         per_fold=ff, F1_oracle=round(prf(*np.sum(list(g[orc].values()), axis=0))[2], 4), deploy=orc))
    R = pd.DataFrame(rows)
    fin = (R.F1_honest * R.cls.map(CLASS_WEIGHTS)).sum() / R.cls.map(CLASS_WEIGHTS).sum()
    return R, fin


base = grid_counts(P); R1, f1 = summarise(base, "base")
D = P[P.lcls.isin(K)].copy(); D["lcls"] = D.lcls.map({K[0]: K[1], K[1]: K[0]}); D["score"] *= 0.8
R2, f2 = summarise(grid_counts(pd.concat([P, D], ignore_index=True)), "dual0.8")
R = pd.concat([R1, R2]); R.to_csv(f"{RUN}/eval_official.csv", index=False)
pd.set_option("display.width", 250); print(R.to_string(index=False))
print(f"\nfinal (satellite classes present, honest): base {f1:.4f}   dual-kurgan {f2:.4f}")
json.dump({r.cls: {"mask": float(r.deploy[0]), "conf": float(r.deploy[1])} for r in R1.itertuples()},
          open(f"{RUN}/deploy_thresholds.json", "w"), indent=1)
