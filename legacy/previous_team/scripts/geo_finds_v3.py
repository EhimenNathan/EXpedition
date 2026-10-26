"""Geophysics point finds v3: tiled candidates + dipole-shape features + rendering augmentation + ensemble.

    python scripts/geo_finds_v3.py extract [--exclude 200_]      # stage A: features per unit and rendering (cached)
    python scripts/geo_finds_v3.py model   [--variants all]      # stage B: leave-one-region-out models + honest eval

Leakage control (strict):
  * evaluation is leave-one-region-out: the held-out region's rows (original rendering only) are scored by models
    trained on the other regions; augmented renderings are TRAINING rows only and always carry their source region;
  * the probability threshold (and any blend choice) for a held-out region is chosen on the other regions' OOF scores;
  * no region identity, no map resolution feature (it identified regions in the object model).
Stage A, per magnetometry unit with finds labels, per rendering (orig + 5 augmentations of the grey value):
  aug_gamma_lo (gamma 0.6), aug_gamma_hi (gamma 1.6), aug_blur (Gaussian 1.5 px: smooth interpolated style),
  aug_contour (posterise to 12 levels + 1 px blur: contour-map style of 211/212), aug_clip (hard 5-95% stretch).
  Tiled (core 1500 px, pad 220 px at 0.1 m) so the 3 cm / 0.5 GB map of region 200 fits in memory.
  Features: v2 physics stack at the candidate + 1 m extremes + rotation-invariant dipole descriptors (radial profile
  of the detrended field at 0.2-1.2 m, directional max/min/asymmetry at 0.4 and 0.8 m, opposite-lobe product).
Stage B variants: base (orig only, single LightGBM = v2 recipe), aug (orig+aug), aug_ens (aug + 3 LightGBM seeds +
  ExtraTrees, probability mean), and the untrained physics score. Output: work/geo/finds_v3/report.csv"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from shapely.geometry import Point
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import geo_data as GD, geo_feats as GF
from koz3.official.fastmatch import soft_match, prf

ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["extract", "model"]); ap.add_argument("--root", default="data_geo")
ap.add_argument("--exclude", default=""); ap.add_argument("--variants", default="all"); ap.add_argument("--gsd", type=float, default=0.1)
a = ap.parse_args()
OUT = "work/geo/finds_v3"; os.makedirs(f"{OUT}/units", exist_ok=True)
GSD = a.gsd; R_POS = 1.5; PRED_R = 0.25; CAND_Q = 0.88; MIN_DIST = 0.5
CORE, PAD = 1500, 220
RENDERS = ["orig", "aug_gamma_lo", "aug_gamma_hi", "aug_blur", "aug_contour", "aug_clip"]


def render(lum, valid, kind, rng):
    """Re-render the map's grey value (0..255 domain) in another plausible display style."""
    if kind == "orig":
        return lum
    v = lum[valid]
    lo, hi = np.percentile(v, [1, 99]) if v.size else (0, 255)
    x = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)
    if kind == "aug_gamma_lo":
        x = x ** 0.6
    elif kind == "aug_gamma_hi":
        x = x ** 1.6
    elif kind == "aug_blur":
        x = ndi.gaussian_filter(x, 1.5)
    elif kind == "aug_contour":
        x = ndi.gaussian_filter(np.round(x * 11) / 11, 1.0)
    elif kind == "aug_clip":
        l2, h2 = np.percentile(x[valid], [5, 95]) if v.size else (0, 1); x = np.clip((x - l2) / max(h2 - l2, 1e-6), 0, 1)
    return (x * 255).astype(np.float32)


def dipole_descriptors(zf, rr, cc, gsd):
    """Rotation-invariant shape of the anomaly around each candidate on the detrended field zf."""
    H, W = zf.shape; F = {}
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for rad in (0.2, 0.4, 0.6, 0.8, 1.2):
        rp = rad / gsd
        ys = np.clip(np.round(rr[:, None] + rp * np.sin(ang)[None]).astype(int), 0, H - 1)
        xs = np.clip(np.round(cc[:, None] + rp * np.cos(ang)[None]).astype(int), 0, W - 1)
        ring = zf[ys, xs]
        F[f"ring_mean_{rad}"] = ring.mean(1)
        if rad in (0.4, 0.8):
            F[f"ring_max_{rad}"] = ring.max(1); F[f"ring_min_{rad}"] = ring.min(1)
            F[f"ring_range_{rad}"] = ring.max(1) - ring.min(1)
            opp = ring[:, :8] * ring[:, 8:]            # product of opposite samples: negative for a dipole
            F[f"opp_prod_{rad}"] = opp.min(1)
            F[f"asym_{rad}"] = np.abs(ring[:, :8] - ring[:, 8:]).max(1) / (np.abs(ring).mean(1) + 1e-3)
    F["centre"] = zf[rr, cc]
    F["centre_minus_ring"] = F["centre"] - F["ring_mean_0.8"]
    return F


def window_cands(lum_w, valid_w, core, gsd):
    """Candidates and features in one window (core = y0, x0, y1, x1 in window pixels)."""
    s = None; dip = None
    for n, ch in GF.iter_channels(lum_w, valid_w, gsd):
        if n in ("log0.15", "log0.3", "log0.6"):
            s = np.abs(ch) if s is None else np.maximum(s, np.abs(ch))
        elif n == "dip0.3":
            dip = ch
    s = s + 0.5 * dip; del dip
    keep = np.zeros_like(valid_w); y0, x0, y1, x1 = core; keep[y0:y1, x0:x1] = True
    s[~(valid_w & keep)] = 0
    return s


def unit_rows(u, kind, rng):
    d = GD.load_unit(u, GSD); gsd = d["gsd"]; valid = d["valid"] & ~d["noise"]
    lum = render(d["lum"], d["valid"], kind, rng)
    H, W = lum.shape
    # pass 1 (tiled): candidate score map; threshold = global quantile over the survey
    score = np.zeros((H, W), np.float32)
    for ty in range(0, H, CORE):
        for tx in range(0, W, CORE):
            wy0, wx0, wy1, wx1 = max(ty - PAD, 0), max(tx - PAD, 0), min(ty + CORE + PAD, H), min(tx + CORE + PAD, W)
            core = (ty - wy0, tx - wx0, min(ty + CORE, H) - wy0, min(tx + CORE, W) - wx0)
            if not valid[ty:ty + CORE, tx:tx + CORE].any():
                continue
            sw = window_cands(lum[wy0:wy1, wx0:wx1], d["valid"][wy0:wy1, wx0:wx1], core, gsd)
            score[ty:min(ty + CORE, H), tx:min(tx + CORE, W)] = sw[core[0]:core[2], core[1]:core[3]]
    score[~valid] = 0
    thr = np.quantile(score[valid], CAND_Q) if valid.any() else np.inf
    pk = peak_local_max(score, min_distance=max(1, int(round(MIN_DIST / gsd))), threshold_abs=thr, exclude_border=False)
    pk = pk[valid[pk[:, 0], pk[:, 1]]]
    rr_all, cc_all = pk[:, 0], pk[:, 1]
    # pass 2 (tiled): sample features at the candidates of each core
    k1 = int(max(3, round(1.0 / gsd))) | 1
    parts = []
    for ty in range(0, H, CORE):
        for tx in range(0, W, CORE):
            sel = (rr_all >= ty) & (rr_all < ty + CORE) & (cc_all >= tx) & (cc_all < tx + CORE)
            if not sel.any():
                continue
            wy0, wx0, wy1, wx1 = max(ty - PAD, 0), max(tx - PAD, 0), min(ty + CORE + PAD, H), min(tx + CORE + PAD, W)
            rr, cc = rr_all[sel] - wy0, cc_all[sel] - wx0
            X = {}
            for n, ch in GF.iter_channels(lum[wy0:wy1, wx0:wx1], d["valid"][wy0:wy1, wx0:wx1], gsd):
                X[n] = ch[rr, cc]
                if n in ("log0.3", "log0.6", "hp2", "z"):
                    X[f"{n}_max1m"] = ndi.maximum_filter(ch, k1)[rr, cc]; X[f"{n}_min1m"] = ndi.minimum_filter(ch, k1)[rr, cc]
                if n == "hp5":
                    X.update(dipole_descriptors(ch, rr, cc, gsd))
            X["cand_score"] = score[rr_all[sel], cc_all[sel]]
            X["row"] = rr_all[sel]; X["col"] = cc_all[sel]
            parts.append(pd.DataFrame(X))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    T = d["T"]; xs = T.c + (df.col.values + 0.5) * T.a; ys = T.f + (df.row.values + 0.5) * T.e
    df["score_rank"] = df.cand_score.rank(pct=True)
    area = max(valid.sum() * gsd * gsd, 1.0); df["cand_density"] = len(df) / area
    if len(df) > 1:
        t = cKDTree(np.c_[xs, ys]); dd, _ = t.query(np.c_[xs, ys], k=2); df["nn_dist"] = dd[:, 1]
        df["n_within_2m"] = [len(v) - 1 for v in t.query_ball_point(np.c_[xs, ys], 2.0)]
    else:
        df["nn_dist"] = 99.0; df["n_within_2m"] = 0
    df["map_std_hp2"] = float(np.std(df.hp2)) if len(df) else 0.0
    df["x"] = xs; df["y"] = ys
    gt = np.array([[p.x, p.y] for p in d["finds"]]) if d["finds"] else np.zeros((0, 2))
    y = np.zeros(len(df), int)
    if len(gt) and len(df):
        t = cKDTree(np.c_[xs, ys]); pairs = []
        for gi, (gx, gy) in enumerate(gt):
            for ci in t.query_ball_point((gx, gy), R_POS):
                pairs.append((np.hypot(xs[ci] - gx, ys[ci] - gy), gi, ci))
        ug, uc = set(), set()
        for dist, gi, ci in sorted(pairs):
            if gi not in ug and ci not in uc:
                ug.add(gi); uc.add(ci); y[ci] = 1
    df["tgt"] = y
    return df.drop(columns=["row", "col"]), gt


def extract():
    rng = np.random.default_rng(0)
    U = [u for u in GD.discover(a.root) if u["method"] == "MagGr" and u["finds"]]
    U = [u for u in U if not (a.exclude and any(u["region"].startswith(e) for e in a.exclude.split(",")))]
    gts = {}
    for u in U:
        uid = f"{u['region']}|{u['key']}"
        for kind in RENDERS:
            p = f"{OUT}/units/{uid.replace('|', '__')}__{kind}.parquet"
            if os.path.exists(p):
                continue
            t0 = time.time()
            try:
                df, gt = unit_rows(u, kind, rng)
            except Exception as e:
                print("FAILED", uid, kind, repr(e)[:160], flush=True); continue
            df["uid"] = uid; df["region"] = u["region"]; df["render"] = kind
            df.to_parquet(p); gts[uid] = gt.tolist()
            print(f"{uid} {kind}: {len(df)} cands, {int(df.tgt.sum())} pos / {len(gt)} finds, {time.time() - t0:.0f}s", flush=True)
    gp = f"{OUT}/gt.json"
    old = json.load(open(gp, encoding="utf-8")) if os.path.exists(gp) else {}
    old.update(gts); json.dump(old, open(gp, "w", encoding="utf-8"), ensure_ascii=False)


def model():
    import lightgbm as lgb
    from sklearn.ensemble import ExtraTreesClassifier
    C = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/units/*.parquet"))], ignore_index=True)
    if a.exclude:
        C = C[~C.region.str.startswith(tuple(a.exclude.split(",")))]
    GT = {k: np.array(v) for k, v in json.load(open(f"{OUT}/gt.json", encoding="utf-8")).items()}
    feats = [c for c in C.columns if c not in ("x", "y", "tgt", "uid", "region", "render")]
    O = C[C.render == "orig"].reset_index(drop=True)
    print(f"orig rows {len(O)}, all rows {len(C)}, regions {O.region.nunique()}, finds {sum(len(GT[u]) for u in O.uid.unique())}, "
          f"features {len(feats)}", flush=True)
    regions = sorted(O.region.unique())
    DIP = [f for f in feats if f.startswith(("ring_", "opp_prod", "asym_", "centre"))]
    VARS = ("phys", "base", "base_nodip", "base_ens", "aug", "aug_ens") if a.variants == "all" else tuple(["phys"] + a.variants.split(","))
    preds = {v: np.full(len(O), np.nan) for v in VARS}
    preds["phys"] = O.groupby("uid").cand_score.rank(pct=True).values
    LGB = lambda seed=0, cs=0.7: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                                                    subsample_freq=1, colsample_bytree=cs, reg_lambda=2.0, random_state=seed, n_jobs=4, verbose=-1)
    for r in regions:
        te = (O.region == r).values
        tr_o = C[(C.region != r) & (C.render == "orig")]; tr_a = C[C.region != r]
        Xo = O.loc[te, feats]
        if "base" in preds:
            preds["base"][te] = LGB().fit(tr_o[feats], tr_o.tgt).predict_proba(Xo)[:, 1]
        if "base_nodip" in preds:
            nd = [f for f in feats if f not in DIP]
            preds["base_nodip"][te] = LGB().fit(tr_o[nd], tr_o.tgt).predict_proba(O.loc[te, nd])[:, 1]
        if "base_ens" in preds:
            ps = [LGB(sd, cs).fit(tr_o[feats], tr_o.tgt).predict_proba(Xo)[:, 1] for sd, cs in ((0, 0.7), (1, 0.5), (2, 0.9))]
            et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.3, n_jobs=4, random_state=0)
            ps.append(et.fit(tr_o[feats].fillna(0), tr_o.tgt).predict_proba(Xo.fillna(0))[:, 1])
            preds["base_ens"][te] = np.mean(ps, axis=0)
        if "aug" in preds or "aug_ens" in preds:
            ps = [LGB(sd, cs).fit(tr_a[feats], tr_a.tgt).predict_proba(Xo)[:, 1] for sd, cs in ((0, 0.7), (1, 0.5), (2, 0.9))]
            if "aug" in preds:
                preds["aug"][te] = ps[0]
            if "aug_ens" in preds:
                et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.3, n_jobs=4, random_state=0)
                ps.append(et.fit(tr_a[feats].fillna(0), tr_a.tgt).predict_proba(Xo.fillna(0))[:, 1])
                preds["aug_ens"][te] = np.mean(ps, axis=0)
        print(f"region {r} done", flush=True)
    for v, p in preds.items():
        O[f"p_{v}"] = p
    O[["uid", "region", "x", "y", "tgt"] + [f"p_{v}" for v in preds]].to_parquet(f"{OUT}/oof.parquet")
    THR = {"phys": [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    for v in [x for x in preds if x != "phys"]:
        THR[v] = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    rows = []
    for v in preds:
        res = {}
        for uid, g in O.groupby("uid"):
            gts = [Point(x, y) for x, y in GT[uid]]
            for t in THR[v]:
                s = g[g[f"p_{v}"] >= t]
                res[(uid, t)] = soft_match([(Point(x, y).buffer(PRED_R, 8), c) for x, y, c in zip(s.x, s.y, s[f"p_{v}"])], gts)[:3]
        uids = sorted(O.uid.unique()); reg = {u: u.split("|")[0] for u in uids}
        tot = np.zeros(3); per = {}
        for r in regions:
            oth = [u for u in uids if reg[u] != r]; mine = [u for u in uids if reg[u] == r]
            best = max(THR[v], key=lambda t: prf(*np.sum([res[(u, t)] for u in oth], axis=0))[2])
            c = np.sum([res[(u, best)] for u in mine], axis=0); tot += c; per[r] = round(prf(*c)[2], 3)
        P_, R_, F_ = prf(*tot)
        rows.append(dict(variant=v, F1_honest=round(F_, 4), P=round(P_, 3), R=round(R_, 3), tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), **per))
        print(f"[{v:8s}] honest F1 {F_:.4f}  P {P_:.3f}  R {R_:.3f}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/report.csv", index=False)
    pd.set_option("display.width", 250); print(R.T.to_string())


if __name__ == "__main__":
    extract() if a.stage == "extract" else model()
