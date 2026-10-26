"""LiDAR kurgan misses (OOF seed ensemble, deployed decoding): heat at each label, per-region recall, and a figure of
the relief (local relief 5 m) around missed vs found kurgans → figures/li_kurgan_misses.jpg."""
import os, sys, pickle
import numpy as np, pandas as pd, cv2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)


def main():
    from koz3.li_train import labels, site_items, decode_site_v2
    from shapely.geometry import shape, Point
    from rasterio.features import rasterize
    from affine import Affine
    from pyproj import Transformer
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = pickle.load(open(os.path.join(ROOT, "work/li/oof_ens.pkl"), "rb"))
    rows, crops = [], []
    for r, p in P.items():
        it = items[r]; meta = dict(it["meta"], region=r); p = p.astype(np.float32)
        feats = decode_site_v2(p, it["valid"], meta, thr=0.15, dual=0.05)
        tr = Transformer.from_crs(3857, meta["crs"], always_xy=True)
        pts = [(Point(*tr.transform(*shape(f["geometry"]).centroid.coords[0])), f["properties"]["class_name"]) for f in feats
               if f["properties"]["class_name"].startswith("kurg")]
        lab = L[(L.region == r) & L.class_name.str.startswith("kurg")].to_crs(meta["crs"])
        T = Affine(meta["res"], 0, meta["x0"], 0, -meta["res"], meta["ytop"])
        x = np.asarray(it["x"], np.float32)
        for g, c in zip(lab.geometry, lab.class_name):
            m = rasterize([(g, 1)], out_shape=p.shape[1:], transform=T).astype(bool)
            if not m.any():
                continue
            hit = any(g.contains(q) for q, _ in pts); same = any(g.contains(q) and cc == c for q, cc in pts)
            heat = float(p[0][m].max())
            rows.append((r, c, hit, same, heat, float(g.area)))
            if len(crops) < 400 and np.random.default_rng(len(rows)).random() < 0.25:
                ys, xs = np.where(m); cy, cx = int(ys.mean()), int(xs.mean()); R = 40
                if R <= cy < x.shape[1] - R and R <= cx < x.shape[2] - R:
                    crops.append((hit, heat, x[1, cy - R:cy + R, cx - R:cx + R].copy(), m[cy - R:cy + R, cx - R:cx + R].copy()))
    D = pd.DataFrame(rows, columns=["region", "cls", "hit", "same", "heat", "area"])
    D.to_parquet(os.path.join(ROOT, "work/li/fn_analysis.parquet"))
    print(len(D), "labels; recall", round(D.hit.mean(), 3))
    g = D.groupby("region").agg(n=("hit", "size"), recall=("hit", "mean"), heat_med=("heat", "median")).sort_values("n", ascending=False)
    print(g.head(12).round(2).to_string())
    print("area median hit/miss", D[D.hit].area.median(), D[~D.hit].area.median())
    tiles = []
    for want in (True, False):
        sel = [c for c in crops if c[0] == want and (want or c[1] < 0.05)][:8]
        row = []
        for _, h, im, m in sel:
            v = np.clip((im - np.percentile(im, 2)) / (np.ptp(im) + 1e-6), 0, 1)
            t = cv2.cvtColor((v * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(t, cnt, -1, (0, 255, 0) if want else (0, 0, 255), 1)
            row.append(cv2.resize(t, (200, 200), interpolation=cv2.INTER_NEAREST))
        while len(row) < 8:
            row.append(np.zeros((200, 200, 3), np.uint8))
        tiles.append(np.hstack(row))
    cv2.imwrite(os.path.join(ROOT, "figures/li_kurgan_misses.jpg"), np.vstack(tiles))


if __name__ == "__main__":
    main()
