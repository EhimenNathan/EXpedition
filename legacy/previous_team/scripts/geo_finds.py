"""Geophysics point finds (class finds_points): candidate detection + LightGBM re-scoring, leave-one-region-out.

    python scripts/geo_finds.py [data_root=data_geo] [--gsd 0.1]

Pipeline per magnetometry unit:
  1. robust-normalised feature stack (koz3/geo_feats.py) at 0.1 m;
  2. candidates = local maxima of an untrained blob/dipole score (min spacing 0.5 m, top 12% of survey pixels),
     restricted to the survey boundary and outside annotated noise areas;
  3. per-candidate features (stack values at the peak, 1 m neighbourhood extremes, map-level context);
  4. target = candidate is the nearest one-to-one match of a labelled find within 1.5 m;
  5. LightGBM trained leave-one-region-out (regions never share a fold);
  6. official-metric evaluation: a find is hit when the predicted centroid is within 2 m; P = TP/(TP+0.5FP).
     Threshold chosen honestly on the other regions' out-of-fold scores.
Writes work/geo/finds_candidates.parquet, work/geo/finds_report.csv and docs/figures/geo_finds_*.png."""
import os, sys, time, json, argparse
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from shapely.geometry import Point
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import geo_data as GD, geo_feats as GF
from koz3.official.fastmatch import soft_match, prf

ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default="data_geo"); ap.add_argument("--gsd", type=float, default=0.1)
ap.add_argument("--cand_q", type=float, default=0.88); ap.add_argument("--min_dist", type=float, default=0.5)
args = ap.parse_args()
os.makedirs("work/geo", exist_ok=True); os.makedirs("docs/figures", exist_ok=True)
R_POS = 1.5      # training target radius (m); evaluation uses the official 2 m
PRED_R = 0.25    # predicted find = small disc around the candidate (only its centroid matters)


def unit_candidates(u):
    d = GD.load_unit(u, args.gsd); gsd = d["gsd"]
    valid = d["valid"] & ~d["noise"]
    # pass 1: untrained candidate score (blob strength of either sign + dipole pairing), keeping only what it needs
    s = None
    for n, ch in GF.iter_channels(d["lum"], d["valid"], gsd):
        if n in ("log0.15", "log0.3", "log0.6"):
            s = np.abs(ch) if s is None else np.maximum(s, np.abs(ch))
        elif n == "dip0.3":
            dip = ch
    s = s + 0.5 * dip; del dip
    s[~valid] = 0
    thr = np.quantile(s[valid], args.cand_q) if valid.any() else np.inf
    pk = peak_local_max(s, min_distance=max(1, int(round(args.min_dist / gsd))), threshold_abs=thr, exclude_border=False)
    pk = pk[valid[pk[:, 0], pk[:, 1]]]
    rr, cc = pk[:, 0], pk[:, 1]
    # pass 2: sample every channel (and 1 m neighbourhood extremes of key channels) at the candidates only
    k = int(max(3, round(1.0 / gsd))) | 1
    X = {}; std_hp2 = 0.0
    for n, ch in GF.iter_channels(d["lum"], d["valid"], gsd):
        X[n] = ch[rr, cc]
        if n in ("log0.3", "log0.6", "hp2", "z"):
            X[f"{n}_max1m"] = ndi.maximum_filter(ch, k)[rr, cc]; X[f"{n}_min1m"] = ndi.minimum_filter(ch, k)[rr, cc]
        if n == "hp2":
            std_hp2 = float(ch[valid].std()) if valid.any() else 0.0
    X["cand_score"] = s[rr, cc]
    X["score_rank"] = pd.Series(s[rr, cc]).rank(pct=True).values
    xs = d["T"].c + (cc + 0.5) * d["T"].a; ys = d["T"].f + (rr + 0.5) * d["T"].e
    area = max(valid.sum() * gsd * gsd, 1.0)
    X["cand_density"] = np.full(len(pk), len(pk) / area)
    if len(pk) > 1:
        t = cKDTree(np.c_[xs, ys]); dd, _ = t.query(np.c_[xs, ys], k=2); X["nn_dist"] = dd[:, 1]
        X["n_within_2m"] = np.array([len(v) - 1 for v in t.query_ball_point(np.c_[xs, ys], 2.0)])
    else:
        X["nn_dist"] = np.full(len(pk), 99.0); X["n_within_2m"] = np.zeros(len(pk))
    X["map_gsd_native"] = np.full(len(pk), gsd)
    X["map_std_z"] = np.full(len(pk), std_hp2)
    X["is_colour"] = np.full(len(pk), float("цвет" in os.path.basename(u["path"])))
    df = pd.DataFrame(X); df["x"] = xs; df["y"] = ys
    # one-to-one target: greedy by distance, within R_POS
    gt = np.array([[p.x, p.y] for p in d["finds"]]) if d["finds"] else np.zeros((0, 2))
    y = np.zeros(len(df), int)
    if len(gt) and len(df):
        t = cKDTree(np.c_[xs, ys]); pairs = []
        for gi, (gx, gy) in enumerate(gt):
            for ci in t.query_ball_point((gx, gy), R_POS):
                pairs.append((np.hypot(xs[ci] - gx, ys[ci] - gy), gi, ci))
        ug, uc = set(), set()
        for dist, gi, ci in sorted(pairs):
            if gi in ug or ci in uc:
                continue
            ug.add(gi); uc.add(ci); y[ci] = 1
    df["tgt"] = y
    # recall ceiling at the official 2 m radius
    ceil = 0
    if len(gt) and len(df):
        dd, _ = cKDTree(np.c_[xs, ys]).query(gt); ceil = int((dd <= 2.0).sum())
    meta = dict(region=u["region"], unit=f"{u['method']}:{u['key']}", img=os.path.basename(u["path"]), n_gt=len(gt),
                n_cand=len(df), ceiling=ceil, area_m2=round(area), gsd=round(gsd, 3))
    return df, gt, d, meta


def main():
    t0 = time.time()
    U = [u for u in GD.discover(args.root) if u["method"] == "MagGr" and u["finds"]]
    frames, GT, metas = [], {}, []
    for u in U:
        try:
            df, gt, d, meta = unit_candidates(u)
        except Exception as e:
            print("skip", u["region"], u["key"], repr(e)); continue
        uid = f"{u['region']}|{u['key']}"; df["uid"] = uid; df["region"] = u["region"]
        frames.append(df); GT[uid] = gt; metas.append(meta)
        print(json.dumps(meta, ensure_ascii=False), f"{time.time() - t0:.0f}s", flush=True)
    C = pd.concat(frames, ignore_index=True); M = pd.DataFrame(metas)
    feats = [c for c in C.columns if c not in ("x", "y", "tgt", "uid", "region")]
    print(f"\n{len(C)} candidates, {int(C.tgt.sum())} positives, {M.n_gt.sum()} finds; recall ceiling (2 m) "
          f"{M.ceiling.sum() / max(M.n_gt.sum(), 1):.3f}", flush=True)

    # leave-one-region-out LightGBM
    import lightgbm as lgb
    regions = sorted(C.region.unique()); C["p"] = np.nan
    imp = np.zeros(len(feats))
    for r in regions:
        tr, te = C.region != r, C.region == r
        if C.loc[tr, "tgt"].sum() < 20:
            continue
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=8, verbose=-1)
        m.fit(C.loc[tr, feats], C.loc[tr, "tgt"]); C.loc[te, "p"] = m.predict_proba(C.loc[te, feats])[:, 1]
        imp += m.booster_.feature_importance("gain")
    C["phys"] = C.groupby("uid").cand_score.rank(pct=True)  # untrained baseline confidence
    C.to_parquet("work/geo/finds_candidates.parquet")

    def evaluate(col, thr_list):
        """Per-region (tp, fp, fn) at each threshold, official 2 m rule."""
        res = {}
        for uid, g in C.groupby("uid"):
            gts = [Point(x, y) for x, y in GT[uid]]
            for t in thr_list:
                s = g[g[col] >= t]
                preds = [(Point(x, y).buffer(PRED_R, 8), c) for x, y, c in zip(s.x, s.y, s[col])]
                res[(uid, t)] = soft_match(preds, gts)[:3]
        return res

    rows = []
    for col, thr_list in (("p", [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]),
                          ("phys", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])):
        res = evaluate(col, thr_list); uids = sorted(C.uid.unique())
        regs = {u: u.split("|")[0] for u in uids}
        # honest: threshold chosen on the other regions
        tot = np.zeros(3); per = []
        for r in sorted(set(regs.values())):
            other = [u for u in uids if regs[u] != r]; mine = [u for u in uids if regs[u] == r]
            best = max(thr_list, key=lambda t: prf(*np.sum([res[(u, t)] for u in other], axis=0))[2])
            c = np.sum([res[(u, best)] for u in mine], axis=0); tot += c
            per.append(dict(model=col, region=r, thr=best, tp=int(c[0]), fp=int(c[1]), fn=int(c[2]), F1=round(prf(*c)[2], 3)))
        P_, R_, F_ = prf(*tot)
        orc = {t: prf(*np.sum([res[(u, t)] for u in uids], axis=0)) for t in thr_list}
        tbest = max(orc, key=lambda t: orc[t][2])
        print(f"\n[{col}] honest: P {P_:.3f} R {R_:.3f} F1 {F_:.3f}   (oracle thr {tbest}: F1 {orc[tbest][2]:.3f})")
        print(pd.DataFrame(per).to_string(index=False))
        rows += per + [dict(model=col, region="ALL(honest)", tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), F1=round(F_, 3))]
    # chance: same number of points as the honest LightGBM output, placed uniformly in each survey
    rng = np.random.default_rng(0); ch = np.zeros(3)
    for uid, g in C.groupby("uid"):
        k = int((g.p >= 0.2).sum()); xs = rng.uniform(g.x.min(), g.x.max(), k); ys = rng.uniform(g.y.min(), g.y.max(), k)
        ch += soft_match([(Point(x, y).buffer(PRED_R, 8), 1.0) for x, y in zip(xs, ys)], [Point(x, y) for x, y in GT[uid]])[:3]
    print(f"\nchance (same count as LGB p>=0.2, uniform in survey bbox): F1 {prf(*ch)[2]:.3f}")
    rows.append(dict(model="chance", region="ALL", tp=int(ch[0]), fp=int(ch[1]), fn=int(ch[2]), F1=round(prf(*ch)[2], 3)))
    pd.DataFrame(rows).to_csv("work/geo/finds_report.csv", index=False); M.to_csv("work/geo/finds_units.csv", index=False)
    fi = pd.Series(imp, feats).sort_values(ascending=False); print("\ntop features:", fi.head(12).round(0).to_dict())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
