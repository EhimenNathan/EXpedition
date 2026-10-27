"""Illumination-consistency features for mound candidates in optical imagery (satellite, aerial film).

Physics: a mound lit by the sun is brighter on the sun-facing slope, so the brightness centroid of a small disc around
it is displaced towards the sun. Every mound in one image shares the sun azimuth; pits, dark soil marks, rings and film
blemishes do not show that displacement (or show it reversed). Rotation / flip augmentation makes CNN detectors blind
to this direction, so the feature is information the detectors cannot carry.

dipole(): per-candidate brightness-centroid displacement (dx east, dy north, normalised).
agreement(): leave-one-out scene azimuth from the other confident candidates of the same image → cosine agreement of
each candidate and the scene concentration R (0 = no common direction, e.g. high sun or ploughed-flat mounds, so a rule
scaled by R switches itself off).
"""
import numpy as np

RADII_M = (4.0, 8.0, 15.0)


def dipole(gray, rows, cols, radius_px):
    """gray (H,W) float; rows/cols candidate pixel centres; radius_px int → (N,2) displacement, NaN off-image."""
    H, W = gray.shape
    R = int(max(2, radius_px))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    m = (xx ** 2 + yy ** 2) <= R * R
    xs, ys = xx[m].astype(np.float64), -yy[m].astype(np.float64)        # +y = north (image rows grow southwards)
    out = np.full((len(rows), 2), np.nan)
    for i, (r, c) in enumerate(zip(np.asarray(rows, int), np.asarray(cols, int))):
        if r - R < 0 or c - R < 0 or r + R + 1 > H or c + R + 1 > W:
            continue
        v = gray[r - R:r + R + 1, c - R:c + R + 1][m].astype(np.float64)
        v = v - v.mean(); s = v.std()
        if not np.isfinite(s) or s < 1e-6:
            continue
        out[i] = ((v * xs).sum(), (v * ys).sum())
        out[i] /= s * m.sum() * R
    return out


def agreement(vec, score, s_min=0.35):
    """vec (N,2) dipoles of one image, score (N,) detector confidence.

    → cos (N,) agreement with the leave-one-out scene azimuth, R (N,) concentration of that azimuth estimate,
      mag (N,) dipole magnitude. Reference candidates: score ≥ s_min (label-free, available at inference).
    """
    vec = np.asarray(vec, float); score = np.asarray(score, float)
    ok = np.isfinite(vec).all(1)
    mag = np.where(ok, np.hypot(vec[:, 0], vec[:, 1]), 0.0)
    u = np.zeros_like(vec); u[ok] = vec[ok] / np.maximum(mag[ok, None], 1e-9)
    ref = ok & (score >= s_min)
    w = np.where(ref, score, 0.0)
    S = (u * w[:, None]).sum(0); Wt = w.sum()
    cos = np.zeros(len(vec)); R = np.zeros(len(vec))
    for i in np.flatnonzero(ok):
        s_i = S - u[i] * w[i]; w_i = Wt - w[i]          # leave this candidate out
        if w_i <= 0:
            continue
        n = np.hypot(*s_i)
        R[i] = n / w_i
        if n > 0:
            cos[i] = float(u[i] @ (s_i / n))
    return cos, R, mag


def scene_features(gray, rows, cols, res_m, score, radii_m=RADII_M, s_min=0.35):
    """All radii; picks per image the radius with the highest mean concentration (label-free). → dict of arrays."""
    best = None
    for rm in radii_m:
        v = dipole(gray, rows, cols, round(rm / res_m))
        cos, R, mag = agreement(v, score, s_min)
        q = float(np.mean(R)) if len(R) else 0.0
        if best is None or q > best[0]:
            best = (q, rm, cos, R, mag)
    _, rm, cos, R, mag = best
    return {"sh_cos": cos, "sh_R": R, "sh_mag": mag, "sh_radius": np.full(len(cos), rm)}
