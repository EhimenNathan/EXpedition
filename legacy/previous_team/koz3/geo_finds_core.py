"""Geophysics point finds: candidate detection + feature extraction shared by training (scripts/geo_finds_v3.py) and
inference (inference/run.py). Identical code path in both, so the deployed model sees exactly the training features.

candidates(d, gsd) -> DataFrame (one row per candidate: features + map coordinates x, y)
  d = koz3.geo_data.load_unit(...) output (lum, valid, noise, T). Tiled (core 1500 px, pad 220 px) for large maps."""
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from . import geo_feats as GF

CAND_Q = 0.88; MIN_DIST = 0.5; CORE, PAD = 1500, 220


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
            F[f"opp_prod_{rad}"] = (ring[:, :8] * ring[:, 8:]).min(1)   # negative for a dipole
            F[f"asym_{rad}"] = np.abs(ring[:, :8] - ring[:, 8:]).max(1) / (np.abs(ring).mean(1) + 1e-3)
    F["centre"] = zf[rr, cc]
    F["centre_minus_ring"] = F["centre"] - F["ring_mean_0.8"]
    return F


def _score_window(lum_w, valid_w, core, gsd):
    s = None; dip = None
    for n, ch in GF.iter_channels(lum_w, valid_w, gsd):
        if n in ("log0.15", "log0.3", "log0.6"):
            s = np.abs(ch) if s is None else np.maximum(s, np.abs(ch))
        elif n == "dip0.3":
            dip = ch
    s = s + 0.5 * dip
    keep = np.zeros_like(valid_w); y0, x0, y1, x1 = core; keep[y0:y1, x0:x1] = True
    s[~(valid_w & keep)] = 0
    return s


def candidates(d, lum=None):
    """Candidates and features for one loaded unit. lum: optional re-rendered grey value (training augmentation)."""
    gsd = d["gsd"]; valid = d["valid"] & ~d["noise"]
    lum = d["lum"] if lum is None else lum
    H, W = lum.shape
    score = np.zeros((H, W), np.float32)
    for ty in range(0, H, CORE):
        for tx in range(0, W, CORE):
            if not valid[ty:ty + CORE, tx:tx + CORE].any():
                continue
            wy0, wx0, wy1, wx1 = max(ty - PAD, 0), max(tx - PAD, 0), min(ty + CORE + PAD, H), min(tx + CORE + PAD, W)
            core = (ty - wy0, tx - wx0, min(ty + CORE, H) - wy0, min(tx + CORE, W) - wx0)
            sw = _score_window(lum[wy0:wy1, wx0:wx1], d["valid"][wy0:wy1, wx0:wx1], core, gsd)
            score[ty:min(ty + CORE, H), tx:min(tx + CORE, W)] = sw[core[0]:core[2], core[1]:core[3]]
    score[~valid] = 0
    if not valid.any():
        return pd.DataFrame()
    thr = np.quantile(score[valid], CAND_Q)
    pk = peak_local_max(score, min_distance=max(1, int(round(MIN_DIST / gsd))), threshold_abs=thr, exclude_border=False)
    pk = pk[valid[pk[:, 0], pk[:, 1]]]
    rr_all, cc_all = pk[:, 0], pk[:, 1]
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
    if not len(df):
        return df
    T = d["T"]; xs = T.c + (df.col.values + 0.5) * T.a; ys = T.f + (df.row.values + 0.5) * T.e
    df["score_rank"] = df.cand_score.rank(pct=True)
    df["cand_density"] = len(df) / max(valid.sum() * gsd * gsd, 1.0)
    if len(df) > 1:
        t = cKDTree(np.c_[xs, ys]); dd, _ = t.query(np.c_[xs, ys], k=2); df["nn_dist"] = dd[:, 1]
        df["n_within_2m"] = [len(v) - 1 for v in t.query_ball_point(np.c_[xs, ys], 2.0)]
    else:
        df["nn_dist"] = 99.0; df["n_within_2m"] = 0
    df["map_std_hp2"] = float(np.std(df.hp2))
    df["x"] = xs; df["y"] = ys
    return df.drop(columns=["row", "col"])
