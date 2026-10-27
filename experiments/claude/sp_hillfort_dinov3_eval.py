"""Stage 3: honest comparison of hillfort localisers over token resolutions (512 / 768 px views) and their multi-scale
ensemble. Probe = PCA(d) fitted on the training folds + L2 logistic (C), out of fold over koz3.sp_prep.site_folds;
settings (d, C, σ) chosen per outer fold on the other folds (nested). Also a fixed, selection-free ensemble (mean of
d ∈ {32, 64, 128} × C ∈ {0.005, 0.05}). HIT = argmax of the smoothed map inside a true hillfort, on scenes that contain
one; scene centre = deployed fallback.
  python experiments/claude/sp_hillfort_dinov3_eval.py work/hf_dinov3/tokens.npz work/hf_dinov3/tokens768.npz
Writes work/hf_dinov3/loc_points_oof.csv (per scene: OOF localiser point in EPSG:3857 + map max) for the pipeline test.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
DIMS, CS, SIGMAS = (32, 64, 128), (0.005, 0.05), (0.5, 1.0)


def oof_maps(Z, tf):
    """{(d, C): out-of-fold token probabilities} for one token file."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tok, y = Z["tok"].astype(np.float32), Z["y"]
    P = {(d, C): np.zeros(len(tok)) for d in DIMS for C in CS}
    for f in np.unique(tf):
        tr, te = tf != f, tf == f
        A = PCA(max(DIMS), random_state=0).fit(tok[tr][::4]).transform(tok)
        for d in DIMS:
            sc = StandardScaler().fit(A[tr, :d])
            for C in CS:
                m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(A[tr, :d]), y[tr])
                P[(d, C)][te] = m.predict_proba(sc.transform(A[te, :d]))[:, 1]
    return P


def per_scene(Z, p, s):
    idx = np.flatnonzero(Z["sid"] == s); gh, gw = Z["grid"][s]
    return idx, p[idx].reshape(gh, gw)


def main(*files):
    import geopandas as gpd
    import pandas as pd
    from scipy.ndimage import gaussian_filter, zoom
    from koz3.sp_prep import site_folds
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]; fo = site_folds(L)
    Zs = [dict(np.load(os.path.join(ROOT, f), allow_pickle=True)) for f in files]
    meta = Zs[0]["meta"]; ns = len(meta)
    assert all(len(z["meta"]) == ns and all(a[1] == b[1] for a, b in zip(z["meta"], meta)) for z in Zs), "scene order differs"
    sf = np.array([fo[m[0]] for m in meta])
    ref = Zs[-1]                                                     # finest grid = decoding grid
    has = np.array([ref["y"][ref["sid"] == s].any() for s in range(ns)]); HS = np.flatnonzero(has)
    maps = {}                                                        # (file k, d, C) → list of per-scene maps on ref grid
    for k, Z in enumerate(Zs):
        P = oof_maps(Z, sf[Z["sid"]])
        for key, p in P.items():
            ms = []
            for s in range(ns):
                _, m = per_scene(Z, p, s); gh, gw = ref["grid"][s]
                ms.append(zoom(m, (gh / m.shape[0], gw / m.shape[1]), order=1) if m.shape != (gh, gw) else m)
            maps[(k,) + key] = ms
        print(f"{files[k]}: out-of-fold maps done", flush=True)

    def hit(ms, sigma):
        out = {}
        for s in HS:
            idx = np.flatnonzero(ref["sid"] == s); g = gaussian_filter(ms[s], sigma, mode="nearest")
            out[s] = bool(ref["y"][idx][int(np.argmax(g))])
        return out
    pos = ref["pos"]
    centre = {}
    for s in HS:
        idx = np.flatnonzero(ref["sid"] == s); gh, gw = ref["grid"][s]
        centre[s] = bool(ref["y"][idx][(gh // 2) * gw + gw // 2])        # token at the scene centre
    print(f"scenes with a hillfort: {len(HS)} | centre token hit {np.mean(list(centre.values())):.3f}")
    avg = lambda keys: [np.mean([maps[k][s] for k in keys], 0) for s in range(ns)]
    families = {f"{os.path.basename(files[k])}": [key for key in maps if key[0] == k] for k in range(len(Zs))}
    if len(Zs) > 1:
        families["multi-scale"] = list(maps)
    H = {}
    for name, keys in families.items():
        for key in keys:
            for sg in SIGMAS:
                H[(name, key, sg)] = hit(maps[key], sg)
        for sg in SIGMAS:
            H[(name, "ens", sg)] = hit(avg(keys), sg)
    for name, keys in families.items():
        cand = [k for k in H if k[0] == name and k[1] != "ens"]
        tot, ch = [], []
        for f in np.unique(sf[HS]):
            best = max(cand, key=lambda k: np.mean([H[k][s] for s in HS if sf[s] != f]))
            tot += [H[best][s] for s in HS if sf[s] == f]; ch.append(best[1:])
        ens = {sg: np.mean(list(H[(name, "ens", sg)].values())) for sg in SIGMAS}
        print(f"[{name}] nested hit {np.mean(tot):.3f} (chosen {ch}) | fixed ensemble σ0.5 {ens[0.5]:.3f} σ1 {ens[1.0]:.3f}")
    # OOF localiser points for the pipeline test: multi-scale (or single) fixed ensemble, σ 0.5
    name = "multi-scale" if len(Zs) > 1 else list(families)[0]
    E = avg(families[name]); rows = []
    for s in range(ns):
        idx = np.flatnonzero(ref["sid"] == s); g = gaussian_filter(E[s], 0.5, mode="nearest"); a = int(np.argmax(g))
        rows.append((meta[s][0], meta[s][1], pos[idx][a, 2], pos[idx][a, 3], float(g.max()), pos[idx][:, 2].mean(), pos[idx][:, 3].mean(),
                     bool(has[s]), bool(ref["y"][idx][a]), sf[s]))
    D = pd.DataFrame(rows, columns=["region", "tif", "x3857", "y3857", "map_max", "cx3857", "cy3857", "has_hillfort", "hit", "fold"])
    D.to_csv(os.path.join(ROOT, "work/hf_dinov3/loc_points_oof.csv"), index=False)
    print(f"saved loc_points_oof.csv ({name} fixed ensemble σ0.5): hit {D[D.has_hillfort].hit.mean():.3f} on {int(D.has_hillfort.sum())} scenes")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["work/hf_dinov3/tokens.npz", "work/hf_dinov3/tokens768.npz"]))
