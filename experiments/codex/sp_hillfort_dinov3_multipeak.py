"""Generate honest multi-peak DINOv3-SAT hillfort candidates.

Uses the same fixed multi-scale probe ensemble as sp_hillfort_dinov3_eval.py.
Every token probability is predicted by a probe trained without its 20 km site
cluster.  Local maxima are suppressed in metric space, not token space, so the
rule is comparable across scene resolutions.
"""
import os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from experiments.claude.sp_hillfort_dinov3_eval import oof_maps, per_scene, DIMS, CS


def main(files):
    import geopandas as gpd
    import pandas as pd
    from scipy.ndimage import gaussian_filter, zoom, maximum_filter
    from koz3.sp_prep import site_folds

    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    L = L[L["mod"] == "SpOr"]
    fo = site_folds(L)
    Zs = [dict(np.load(os.path.join(ROOT, f), allow_pickle=True)) for f in files]
    ref, meta = Zs[-1], Zs[0]["meta"]
    sf = np.array([fo[m[0]] for m in meta])
    all_maps = []
    for k, Z in enumerate(Zs):
        P = oof_maps(Z, sf[Z["sid"]])
        for p in P.values():
            ms = []
            for s in range(len(meta)):
                _, m = per_scene(Z, p, s); gh, gw = ref["grid"][s]
                ms.append(zoom(m, (gh / m.shape[0], gw / m.shape[1]), order=1)
                          if m.shape != (gh, gw) else m)
            all_maps.append(ms)
        print(f"{files[k]}: {len(P)} OOF maps", flush=True)

    rows = []
    for s in range(len(meta)):
        idx = np.flatnonzero(ref["sid"] == s)
        g = gaussian_filter(np.mean([m[s] for m in all_maps], axis=0), 0.5, mode="nearest")
        # 3x3 local maxima; later suppress by true EPSG:3857 distance.
        cand = np.argwhere((g == maximum_filter(g, size=3, mode="nearest")))
        cand = sorted(cand, key=lambda q: float(g[tuple(q)]), reverse=True)
        pos = ref["pos"][idx]
        keep = []
        for yy, xx in cand:
            a = int(yy * g.shape[1] + xx); X, Y = map(float, pos[a, 2:4])
            if all(np.hypot(X-x, Y-y) >= 35.0 for x, y, _ in keep):
                keep.append((X, Y, float(g[yy, xx])))
            if len(keep) >= 8:
                break
        mx = keep[0][2]
        for rank, (X, Y, score) in enumerate(keep, 1):
            rows.append((meta[s][0], meta[s][1], float(pos[:,2].mean()), float(pos[:,3].mean()),
                         X, Y, score, score / max(mx, 1e-9), rank, int(sf[s])))
    out = os.path.join(ROOT, "work/hf_dinov3/loc_multipeak_oof.csv")
    pd.DataFrame(rows, columns=["region","tif","cx3857","cy3857","x3857","y3857","map_score","ratio","rank","fold"]).to_csv(out,index=False)
    print(f"saved {len(rows)} candidates to {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["work/hf_dinov3/tokens.npz", "work/hf_dinov3/tokens768.npz"])
