"""Satellite re-scorer: LightGBM over D-FINE + U-Net candidates, out-of-fold on the same 20 km grouped folds.

    python scripts/stack_sp.py [--det work/sp_runs/sp_dfine_l] [--unet work/sp_runs/sp_r34_g] [--unet_mask 0.3]
                               [--out work/sp_runs/stack_v1] [--fig]

1. Candidates: every detector box (score >= 0.05) + U-Net candidates at --unet_mask whose representative point is not
   inside a same-class detector box. Kurgan candidates are offered as BOTH kurgan classes (swap flag), so the model
   also learns the intact/damaged call; other classes keep their own class.
2. Features (per candidate x class): own score and source; best detector / U-Net score on the spot for this class and
   for any other class; the detector's best kurgan score of the other kurgan class; log area, elongation; detector boxes
   >= 0.3 within 50 m; scene count of detector boxes >= 0.5 and scene median of the top-20 detector scores.
3. Target: the candidate would hit a label of that class under the official rule (cov >= 0.5 of the smaller shape AND
   representative point inside the label).
4. LightGBM trained on 3 folds, predicts the 4th (grouped, so every score is out-of-fold).
5. Dedupe per scene and class group (both kurgan classes together): keep candidates in descending new score unless
   their representative point lies inside an already-kept polygon of the group (or vice versa).
6. One shared cut-off per outer fold, chosen on the OTHER folds' out-of-fold scores; official pooled scoring per fold.
Writes <out>/cands.parquet (features, oof score), <out>/report.json; prints the comparison with the detector alone."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from shapely import wkb, STRtree
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from koz3.official.fastmatch import soft_match, prf
from koz3.official.compute_metrics import CLASS_WEIGHTS, normalize_class_name

ap = argparse.ArgumentParser()
ap.add_argument("--det", default="work/sp_runs/sp_dfine_l"); ap.add_argument("--unet", default="work/sp_runs/sp_r34_g")
ap.add_argument("--unet_mask", type=float, default=0.3); ap.add_argument("--out", default="work/sp_runs/stack_v1")
ap.add_argument("--det_conf", type=float, default=0.5); ap.add_argument("--reuse", action="store_true")  # re-score saved candidates
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
K = ("kurgany_tselye", "kurgany_povrezhdennye"); t0 = time.time()


def load(d, src):
    P = pd.read_parquet(f"{d}/oof_pred.parquet"); P["lcls"] = P.cls.map(normalize_class_name); P["src"] = src
    return P


D = load(a.det, "det"); U = load(a.unet, "unet"); U = U[np.isclose(U.thr, a.unet_mask)]
G = pd.read_parquet(f"{a.det}/oof_gt.parquet"); G["lcls"] = G.cls.map(normalize_class_name)
for X in (D, U, G):
    X["g"] = [wkb.loads(x) for x in X.wkb]
fold = {**dict(zip(G.scene, G.fold)), **dict(zip(D.scene, D.fold)), **dict(zip(U.scene, U.fold))}
print(f"det {len(D)} unet {len(U)} gt {len(G)}  {time.time() - t0:.0f}s", flush=True)


def group_of(c):
    return "kurgan" if c in K else c


rows = []
for s in ([] if a.reuse else sorted(set(D.scene) | set(U.scene) | set(G.scene))):
    ds, us, gs = D[D.scene == s], U[U.scene == s], G[G.scene == s]
    dg, ug = list(ds.g), list(us.g)
    dt = STRtree(dg) if dg else None; ut = STRtree(ug) if ug else None
    d_sc, d_cl = ds.score.values, ds.lcls.values; u_sc, u_cl = us.score.values, us.lcls.values
    gt_by = {c: list(gs[gs.lcls == c].g) for c in set(gs.lcls)}
    gtt = {c: STRtree(v) for c, v in gt_by.items()}
    top = np.sort(d_sc)[::-1][:20]; scene_med = float(np.median(top)) if len(top) else 0.0; scene_n = int((d_sc >= 0.5).sum())
    d05 = ds[ds.score >= 0.3]; dct = STRtree(list(d05.g)) if len(d05) else None
    cover = np.zeros(len(us), bool)  # U-Net candidates already covered by a same-class detector box
    for i, (g, c) in enumerate(zip(ug, u_cl)):
        if dt is not None:
            hit = dt.query(g.representative_point(), predicate="intersects")
            cover[i] = any(d_cl[j] == c for j in hit)
    cands = [(g, sc, c, "det") for g, sc, c in zip(dg, d_sc, d_cl)] + [(g, sc, c, "unet") for g, sc, c, cv in zip(ug, u_sc, u_cl, cover) if not cv]
    for g, own, c0, src in cands:
        rp = g.representative_point()
        hd = dt.query(rp, predicate="intersects") if dt is not None else []
        hu = ut.query(rp, predicate="intersects") if ut is not None else []
        dens = len(dct.query(rp.buffer(50.0))) if dct is not None else 0
        x0, y0, x1, y1 = g.bounds; el = max(x1 - x0, y1 - y0) / max(min(x1 - x0, y1 - y0), 1e-6)
        for c in (K if c0 in K else (c0,)):
            other_k = K[1] if c == K[0] else K[0]
            f = dict(scene=s, fold=fold.get(s), cls=c, src=int(src == "det"), swapped=int(c != c0), own=own,
                     d_same=max([d_sc[j] for j in hd if d_cl[j] == c], default=0.0),
                     u_same=max([u_sc[j] for j in hu if u_cl[j] == c], default=0.0),
                     d_other=max([d_sc[j] for j in hd if d_cl[j] != c], default=0.0),
                     u_other=max([u_sc[j] for j in hu if u_cl[j] != c], default=0.0),
                     d_otherk=max([d_sc[j] for j in hd if d_cl[j] == other_k], default=0.0) if c in K else 0.0,
                     larea=float(np.log1p(g.area)), elong=float(min(el, 20)), dens50=dens, scene_n=scene_n, scene_med=scene_med,
                     wkb=g.wkb)
            y = 0
            if c in gtt:
                for j in gtt[c].query(rp, predicate="intersects"):
                    G_ = gt_by[c][j]; inter = g.intersection(G_).area
                    if inter / max(min(g.area, G_.area), 1e-9) >= 0.5:
                        y = 1; break
            f["y"] = y; rows.append(f)
C = pd.read_parquet(f"{a.out}/cands.parquet") if a.reuse else pd.DataFrame(rows); C["cls_i"] = C.cls.map({c: i for i, c in enumerate(CLASS_WEIGHTS)})
print(f"candidates x class: {len(C)}  positives {int(C.y.sum())}  {time.time() - t0:.0f}s", flush=True)

import lightgbm as lgb
FEATS = ["cls_i", "src", "swapped", "own", "d_same", "u_same", "d_other", "u_other", "d_otherk", "larea", "elong", "dens50", "scene_n", "scene_med"]
FEATS = [f for f in FEATS if f not in os.environ.get("KOZ3_DROP", "").split(",")]  # ablation: drop features
if os.environ.get("KOZ3_DET_ONLY"):  # ablation: detector candidates only, no kurgan swap
    C = C[(C.src == 1) & (C.swapped == 0)].reset_index(drop=True)
if not a.reuse:
    C["oof"] = np.nan
for f in ([] if (a.reuse and not os.environ.get("KOZ3_RETRAIN")) else sorted(C.fold.dropna().unique())):
    tr, te = C.fold != f, C.fold == f
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    m.fit(C.loc[tr, FEATS], C.loc[tr, "y"], categorical_feature=["cls_i"])
    C.loc[te, "oof"] = m.predict_proba(C.loc[te, FEATS])[:, 1]
imp = {} if (a.reuse and not os.environ.get("KOZ3_RETRAIN")) else dict(zip(FEATS, m.booster_.feature_importance("gain").round(0).tolist()))
print("feature gain (last fold):", imp, flush=True)


def dedupe(P):
    """per scene and class group, descending score; drop a candidate whose rep point is in a kept one (or vice versa)."""
    keep = []
    for (s, grp), idx in P.groupby([P.scene, P.cls.map(group_of)]).groups.items():
        sub = P.loc[idx].sort_values("score", ascending=False); kept = []
        for i, r in sub.iterrows():
            g = r.g; rp = g.representative_point()
            if any(k.contains(rp) or g.contains(k.representative_point()) for k in kept):
                continue
            kept.append(g); keep.append(i)
    return P.loc[keep]


def final_by_fold(P, thr):
    """official pooled per class -> class-weighted final, per fold and overall."""
    P = P[P.score >= thr]; res = {}
    for f in sorted(G.fold.unique()):
        Pf, Gf = P[P.fold == f], G[G.fold == f]; num = den = 0.0
        for c in CLASS_WEIGHTS:
            Gc = Gf[Gf.lcls == c]
            if not len(G[G.lcls == c]):
                continue
            tot = np.zeros(3)
            for s in set(Gc.scene) | set(Pf[Pf.cls == c].scene):
                pc = Pf[(Pf.scene == s) & (Pf.cls == c)]
                tp, fp, fn, _ = soft_match(list(zip(pc.g, pc.score)), list(Gc[Gc.scene == s].g)); tot += (tp, fp, fn)
            num += prf(*tot)[2] * CLASS_WEIGHTS[c]; den += CLASS_WEIGHTS[c]
        res[f] = num / den
    return res


C["g"] = [wkb.loads(x) for x in C.wkb]
if os.environ.get("KOZ3_DIAG"):  # does the pipeline itself (dedupe across kurgan classes) cost score? detector only, no model
    D0 = C[(C.src == 1) & (C.swapped == 0)].assign(score=lambda x: x.own)
    for nm, P0 in (("detector, no dedupe", D0), ("detector + dedupe", dedupe(D0))):
        r = final_by_fold(P0, a.det_conf); print(f"DIAG {nm:22s} " + "  ".join(f"{r[f]:.3f}" for f in sorted(r)) + f"   {np.mean(list(r.values())):.4f}", flush=True)
    sys.exit()
S = dedupe(C.assign(score=C.oof))
grid = [float(x) for x in os.environ.get("KOZ3_GRID", "0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1,0.12,0.15,0.2").split(",")]  # calibrated P(hit): break-even ~ F/(4-F) ~ 0.07
by_thr = {t: final_by_fold(S, t) for t in grid}
honest, chosen = {}, {}
for f in sorted(G.fold.unique()):  # cut-off chosen on the other folds
    t = max(grid, key=lambda t: np.mean([v for k, v in by_thr[t].items() if k != f])); chosen[f] = t; honest[f] = by_thr[t][f]
Dbase = D[D.score >= a.det_conf].assign(cls=lambda x: x.lcls)
base = final_by_fold(Dbase, a.det_conf)
print("\nper fold          " + "  ".join(f"f{f}" for f in sorted(G.fold.unique())) + "   mean")
print("D-FINE >= %.2f    " % a.det_conf + "  ".join(f"{base[f]:.3f}" for f in sorted(base)) + f"   {np.mean(list(base.values())):.4f}")
print("stack (honest)    " + "  ".join(f"{honest[f]:.3f}" for f in sorted(honest)) + f"   {np.mean(list(honest.values())):.4f}   cut-offs {chosen}")
for t in grid:
    print(f"  stack @ {t:.2f}     " + "  ".join(f"{by_thr[t][f]:.3f}" for f in sorted(by_thr[t])) + f"   {np.mean(list(by_thr[t].values())):.4f}")
if not a.reuse:
    C.drop(columns=["g"]).to_parquet(f"{a.out}/cands.parquet") if not os.environ.get("KOZ3_RETRAIN") else None
S.drop(columns=["g"]).to_parquet(f"{a.out}/oof_pred_dedup{os.environ.get('KOZ3_TAG', '')}.parquet")
js = lambda d: {str(k): float(v) for k, v in d.items()}
json.dump(dict(base=js(base), honest=js(honest), chosen=js(chosen), by_thr={str(k): js(v) for k, v in by_thr.items()}, gain=imp, feats=FEATS),
          open(f"{a.out}/report{os.environ.get('KOZ3_TAG', '')}.json", "w"), indent=1, default=float)
print(f"done {time.time() - t0:.0f}s")
