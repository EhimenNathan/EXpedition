"""Satellite U-Net vs D-FINE per scene, and fixed-rule fusions, under the official metric (koz3-1.5.0, fastmatch).

    python scripts/compare_fuse_sp.py [--unet work/sp_runs/sp_r34_g] [--det work/sp_runs/sp_dfine_l] [--det_conf 0.5] [--low 0.3]

Each run at its deployed policy (U-Net: deploy_thresholds.json per class; detector: one shared confidence).
Fusions (fixed rules, no tuning, so every number is out-of-fold):
  union    = detector + U-Net candidates whose representative point is not inside a same-class detector box;
  verified = detector >= det_conf, plus detector candidates in [low, det_conf) whose representative point lies inside
             a same-class U-Net candidate (the U-Net acts as an independent second opinion for low-confidence boxes).
Prints the pooled final score (class-weighted mean over classes present), per-class F1, per-fold finals, and how many
scenes each model wins on true positives."""
import argparse, json, os, sys
import numpy as np, pandas as pd
from shapely import wkb, STRtree
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from koz3.official.fastmatch import soft_match, prf
from koz3.official.compute_metrics import CLASS_WEIGHTS, normalize_class_name

ap = argparse.ArgumentParser(); ap.add_argument("--unet", default="work/sp_runs/sp_r34_g"); ap.add_argument("--det", default="work/sp_runs/sp_dfine_l")
ap.add_argument("--det_conf", type=float, default=0.5); ap.add_argument("--low", type=float, default=0.3); a = ap.parse_args()


def load(d):
    P = pd.read_parquet(f"{d}/oof_pred.parquet"); P["lcls"] = P.cls.map(normalize_class_name); P["g"] = [wkb.loads(x) for x in P.wkb]
    return P


U = load(a.unet); th = json.load(open(f"{a.unet}/deploy_thresholds.json", encoding="utf-8"))
U = pd.concat([U[(U.lcls == c) & np.isclose(U.thr, v["mask"]) & (U.score >= v["conf"])] for c, v in th.items()])
D = load(a.det)
G = pd.read_parquet(f"{a.det}/oof_gt.parquet"); G["lcls"] = G.cls.map(normalize_class_name); G["g"] = [wkb.loads(x) for x in G.wkb]
fold = dict(zip(G.scene, G.fold)); fold.update(dict(zip(D.scene, D.fold)))


def inside_any(P, Q):
    """mask over P: representative point inside a same-scene, same-class geometry of Q."""
    out = np.zeros(len(P), bool)
    for (s, c), idx in P.groupby(["scene", "lcls"]).groups.items():
        q = Q[(Q.scene == s) & (Q.lcls == c)]
        if not len(q):
            continue
        t = STRtree(list(q.g))
        for k, i in enumerate(idx):
            out[P.index.get_loc(i)] = len(t.query(P.at[i, "g"].representative_point(), predicate="within")) > 0 or \
                len(t.query(P.at[i, "g"].representative_point(), predicate="intersects")) > 0
    return out


def score(P, name):
    rows, per_scene = {}, {}
    for c in CLASS_WEIGHTS:
        Gc = G[G.lcls == c]
        if not len(Gc):
            continue
        tot = np.zeros(3); fz = {}
        for s in set(Gc.scene) | set(P[P.lcls == c].scene):
            pc = P[(P.scene == s) & (P.lcls == c)]
            tp, fp, fn, _ = soft_match(list(zip(pc.g, pc.score)), list(Gc[Gc.scene == s].g))
            tot += (tp, fp, fn); fz.setdefault(fold.get(s), np.zeros(3)); fz[fold.get(s)] += (tp, fp, fn)
            per_scene[s] = per_scene.get(s, 0) + tp
        rows[c] = (prf(*tot)[2], fz)
    w = sum(CLASS_WEIGHTS[c] for c in rows)
    final = sum(rows[c][0] * CLASS_WEIGHTS[c] for c in rows) / w
    folds = sorted({f for c in rows for f in rows[c][1]})
    pf = [sum(prf(*rows[c][1].get(f, np.zeros(3)))[2] * CLASS_WEIGHTS[c] for c in rows) / w for f in folds]
    print(f"{name:34s} final {final:.4f}  folds {np.round(pf, 3).tolist()}  " + "  ".join(f"{c[:8]} {v[0]:.3f}" for c, v in rows.items()))
    return final, per_scene


D_hi = D[D.score >= a.det_conf]
fu, su = score(U, "U-Net R34 (deploy)")
fd, sd = score(D_hi, f"D-FINE >= {a.det_conf}")
scn = sorted(set(su) | set(sd)); wu = sum(su.get(s, 0) > sd.get(s, 0) for s in scn); wd = sum(sd.get(s, 0) > su.get(s, 0) for s in scn)
print(f"scenes where U-Net finds more TPs: {wu}; D-FINE more: {wd}; equal: {len(scn) - wu - wd}")
U2 = U.reset_index(drop=True); D2 = D_hi.reset_index(drop=True)
union = pd.concat([D2, U2[~inside_any(U2, D2)]], ignore_index=True)
score(union, "union (D-FINE + uncovered U-Net)")
mid = D[(D.score >= a.low) & (D.score < a.det_conf)].reset_index(drop=True)
verified = pd.concat([D2, mid[inside_any(mid, U2)]], ignore_index=True)
score(verified, f"verified (D-FINE >= {a.det_conf} + U-Net-confirmed {a.low}-{a.det_conf})")
