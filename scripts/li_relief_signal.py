"""Is the signal there? Per labelled LiDAR kurgan: relief prominence (max of the detrended DTM channels inside the
polygon, metres) vs the deployed CNN blend's max heat inside the polygon. Per site: CNN recall vs relief-visible share.
Background: local maxima of the 10 m detrended relief outside labels (density per km²) → precision / recall a pure
classical relief-peak detector would reach at each prominence threshold, CNN-missed kurgans counted separately.
  python scripts/li_relief_signal.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    from scipy.ndimage import maximum_filter
    from rasterio.features import rasterize
    from koz3.li_train import labels, site_items
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    rows, bg = [], []
    for r in sorted(P["ens"]):
        it = items[r]; x = it["x"]; res = it["meta"]["res"]
        heat = (P["ens"][r][0].astype(np.float32) + P["aug"][r][0] + P["augx"][r][0]) / 3
        rel5, rel10 = np.asarray(x[1], np.float32) * 0.5, np.asarray(x[2], np.float32) * 0.5
        H, W = heat.shape
        objs = [(c, g) for c, g in it["objs"] if c in (0, 1)]
        lab = np.zeros((H, W), np.int32)
        if objs:
            lab = rasterize([(g.buffer(6 / res), i + 1) for i, (c, g) in enumerate(objs)], out_shape=(H, W), dtype="int32")
        for i, (c, g) in enumerate(objs):
            m = rasterize([(g, 1)], out_shape=(H, W), dtype="uint8").astype(bool)
            if not m.any():
                cy, cx = int(g.centroid.y), int(g.centroid.x)
                if not (0 <= cy < H and 0 <= cx < W):
                    continue
                m[cy, cx] = True
            rows.append((r, c, g.area * res * res, float(rel5[m].max()), float(rel10[m].max()), float(heat[m].max())))
        v = np.asarray(it["valid"], bool)
        pk = (rel10 == maximum_filter(rel10, size=int(round(8 / res)))) & v & (rel10 > 0.1) & (lab == 0)
        area_km2 = v.sum() * res * res / 1e6
        bg.append((r, area_km2, *[int((pk & (rel10 >= t)).sum()) for t in (0.1, 0.2, 0.3, 0.5, 0.8)]))
        print(r, len(objs), f"{area_km2:.2f} km²", flush=True)
    D = pd.DataFrame(rows, columns=["site", "cls", "area", "rel5", "rel10", "heat"])
    B = pd.DataFrame(bg, columns=["site", "km2", "b0.1", "b0.2", "b0.3", "b0.5", "b0.8"])
    D.to_parquet("work/li_relief_signal.parquet"); B.to_parquet("work/li_relief_bg.parquet")
    D["missed"] = D.heat < 0.15
    print(f"\n{len(D)} labelled kurgans; CNN-missed (heat < 0.15): {D.missed.sum()}")
    print("relief prominence (10 m detrend, m) quantiles — detected vs missed:")
    print(D.groupby("missed").rel10.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
    print("\nshare with prominence ≥ t:")
    for t in (0.1, 0.2, 0.3, 0.5, 0.8):
        nb = B[f"b{t}"].sum(); km = B.km2.sum()
        print(f"  t {t}: detected {np.mean(D.rel10[~D.missed] >= t):.2f}  missed {np.mean(D.rel10[D.missed] >= t):.2f}"
              f"  | background peaks ≥ t: {nb} ({nb / km:.0f}/km²) vs missed-kurgans ≥ t: {int((D.rel10[D.missed] >= t).sum())}")
    s = D.groupby("site").agg(n=("heat", "size"), cnn_recall=("missed", lambda m: 1 - m.mean()), rel10_med=("rel10", "median"),
                              vis03=("rel10", lambda v: (v >= 0.3).mean()))
    s = s.join(B.set_index("site")[["km2", "b0.3"]]); s["bg03_per_km2"] = s["b0.3"] / s.km2
    print("\nper site (n ≥ 10):"); print(s[s.n >= 10].sort_values("cnn_recall").round(2).to_string())


if __name__ == "__main__":
    main()
