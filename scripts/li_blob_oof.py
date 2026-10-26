"""Physics-informed LiDAR mound detector (site-independent candidates): local maxima of the 10 m local-relief model
(smoothed, height ≥ ~7 cm above trend) → physical morphometry features (relief at 2/5/10/25 m, rings 1–12 m, ring
asymmetry, slope, curvature, summit dent, ditch, height) → LightGBM trained on the OTHER folds (nested). Variant "+cnn"
also sees the CNN heat / mask probabilities. Kurgan candidates far (> 4 m) from every CNN kurgan peak are ADDED to the
deployed CNN decoding; thresholds chosen on the other folds. Per-site recall for the sites the CNN does not transfer to.

  python scripts/li_blob_oof.py [work/li/oof_ens.pkl]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from li_kurgan_cls import feats_at  # noqa: E402

K = ("kurgany_povrezhdennye", "kurgany_tselye")


def candidates(x, valid, min_h=0.15, nms_px=6):
    """Local maxima of smoothed lrm10 (channel 2 = (z − g10)/0.5 → 0.15 ≈ 7.5 cm) inside valid data."""
    s = gaussian_filter(x[2].astype(np.float32), 1.5)
    pk = (s == maximum_filter(s, size=2 * nms_px + 1)) & (s >= min_h) & valid
    pk[:12, :] = pk[-12:, :] = pk[:, :12] = pk[:, -12:] = False
    r, c = np.where(pk)
    return r, c, s[r, c]


def main(oof="work/li/oof_ens.pkl"):
    import lightgbm as lgb
    from pyproj import Transformer
    from shapely import STRtree
    from shapely.geometry import Point, shape
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    from koz3.score import score
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = pickle.load(open(os.path.join(ROOT, oof), "rb"))
    cache = os.path.join(ROOT, "work/li/blob_cands.parquet")
    deployed = {r: decode_site_v2(P[r].astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=0.15, dual=0.05) for r in P}
    if os.path.exists(cache) and os.environ.get("REBUILD") != "1":
        D = pd.read_parquet(cache)
    else:
        rows = []
        for r, p in P.items():
            it = items[r]; meta = it["meta"]; x = np.asarray(it["x"], np.float32); p = p.astype(np.float32)
            rr, cc, h = candidates(x, it["valid"])
            if len(rr) > 6000:                                  # rough terrain: keep the 6000 highest bumps
                o = np.argsort(-h)[:6000]; rr, cc, h = rr[o], cc[o], h[o]
            tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True)
            X = meta["x0"] + (cc + 0.5) * meta["res"]; Y = meta["ytop"] - (rr + 0.5) * meta["res"]
            mx, my = tr.transform(X, Y)
            cnn = [shape(f["geometry"]).centroid.coords[0] for f in deployed[r] if f["properties"]["class_name"] in K]
            dcnn = cKDTree(np.array(cnn)).query(np.c_[mx, my])[0] if cnn else np.full(len(rr), 1e9)
            lab = L[(L.region == r) & L.class_name.isin(K)].to_crs(meta["crs"])
            geoms = list(lab.geometry); tree = STRtree(geoms) if geoms else None
            best = {}; cands = []
            for i in range(len(rr)):
                d = feats_at(x, p, int(rr[i]), int(cc[i]))
                d.update(region=r, fold=it["fold"], x=float(mx[i]), y=float(my[i]), h=float(h[i]), d_cnn=float(dcnn[i]), lab=0)
                if tree is not None:
                    pt = Point(X[i], Y[i])
                    for j in tree.query(pt):
                        if geoms[j].contains(pt) and h[i] > best.get(j, (-1, -1))[0]:
                            best[j] = (h[i], len(cands))
                cands.append(d)
            for _, i in best.values():
                cands[i]["lab"] = 1
            xy = np.array([[c["x"], c["y"]] for c in cands]) if cands else np.zeros((0, 2))
            if len(xy):
                t = cKDTree(xy)
                for R in (10, 25, 60):
                    for i, n in enumerate(t.query_ball_point(xy, R)):
                        cands[i][f"nb{R}"] = len(n) - 1
            rows += cands
            print(r, len(cands), "candidates", sum(c["lab"] for c in cands), "on kurgans /", len(geoms), flush=True)
        D = pd.DataFrame(rows); D.to_parquet(cache)
    print(len(D), "candidates;", int(D.lab.sum()), "best matches of kurgan labels")
    phys = [c for c in D.columns if c not in ("region", "fold", "x", "y", "lab", "d_cnn") and not c.startswith(("p_", "q_", "z_")) and c != "peak"]
    # test-time site adaptation (no labels): each bump relative to its own site's bump population
    g = D.groupby("region")
    for c in phys:
        med = g[c].transform("median"); iqr = g[c].transform(lambda v: v.quantile(0.9) - v.quantile(0.1)) + 1e-6
        D[f"z_{c}"] = (D[c] - med) / iqr
    D["z_rank_h"] = g["h"].rank(pct=True)
    zfeat = [f"z_{c}" for c in phys] + ["z_rank_h"]
    withcnn = phys + ["p_dam", "p_int", "p_ratio", "peak"]
    from sklearn.metrics import roc_auc_score
    for name, cols in (("phys", phys), ("physcnn", withcnn), ("physz", phys + zfeat), ("physzcnn", withcnn + zfeat)):
        D[f"q_{name}"] = np.nan
        for f in sorted(D.fold.unique()):
            tr_ = D[D.fold != f]
            m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=0, verbose=-1)
            m.fit(tr_[cols], tr_.lab)
            D.loc[D.fold == f, f"q_{name}"] = m.predict_proba(D.loc[D.fold == f, cols])[:, 1]
        print(f"{name:9s} AUC {roc_auc_score(D.lab, D[f'q_{name}']):.3f}")
    print(f"CNN heat AUC on the same candidates {roc_auc_score(D.lab, D.peak):.3f}")
    D.to_parquet(os.path.join(ROOT, "work/li/blob_oof.parquet"))
    folds = sorted(D.fold.unique())
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if items[r]["fold"] == f])]) for f in folds}
    kk = lambda yy: 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(yy / 6378137.0)) - np.pi / 2)))

    def emit(G, src):
        out = []
        for row in G.itertuples():
            pdam = row.p_ratio
            e = [("kurgany_povrezhdennye", pdam), ("kurgany_tselye", 1 - pdam)]; e.sort(key=lambda z: -z[1])
            e = [e[0]] + [z for z in e[1:] if z[1] >= 0.05]
            for c, pc in e:
                out.append({"type": "Feature", "geometry": Point(row.x, row.y).buffer(kk(row.y), 12).__geo_interface__,
                            "properties": {"class_name": c, "confidence": float(getattr(row, src) * pc)}})
        return out

    def make(s, f):
        src, t = s
        if src.startswith("R:"):                 # replace: CNN kurgans dropped, physics candidates emitted
            out = [g for r in P if items[r]["fold"] == f for g in deployed[r] if g["properties"]["class_name"] not in K]
            return {"type": "FeatureCollection", "features": out + emit(D[(D.fold == f) & (D[src[2:]] >= t)], src[2:])}
        out = [g for r in P if items[r]["fold"] == f for g in deployed[r]]
        if src != "deployed":
            out += emit(D[(D.fold == f) & (D[src] >= t) & (D.d_cnn > 4.0)], src)
        return {"type": "FeatureCollection", "features": out}
    S = [("deployed", 0)] + [(s_, t) for s_ in ("R:q_phys", "R:q_physcnn", "R:q_physz", "R:q_physzcnn") for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)]
    C = fold_counts(make, S, folds, gt, CLS)
    for src in ("deployed", "R:q_phys", "R:q_physcnn", "R:q_physz", "R:q_physzcnn"):
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[0] == src})
        rec = []
        for r in ("021_Новенькое", "011_Эрентуна", "036_Тимерево", "102_Гнёздово_Заольшанская", "030_Эсбю"):
            f = items[r]["fold"]; s_ = ch[f]
            fs = deployed[r] if s_[0] == "deployed" else [g for g in deployed[r] if g["properties"]["class_name"] not in K] + emit(D[(D.region == r) & (D[s_[0][2:]] >= s_[1])], s_[0][2:])
            _, res = score({"type": "FeatureCollection", "features": fs}, to_gt_geojson(L[L.region == r]))
            k = [res[c] for c in K if c in res]; tp, fn = sum(v["tp"] for v in k), sum(v["fn"] for v in k)
            rec.append(f"{r[:8]} {tp / max(tp + fn, 1):.2f}")
        print(f"{src:11s} honest {h:.4f} (oracle {o:.4f} @ {os_[1]}) chosen {[c[1] for c in ch.values()]} "
              + " ".join(f"{c[:8]} {2 * v[0] / max(2 * v[0] + v[2] + 0.5 * v[1], 1):.3f}" for c, v in tot.items() if v[0] + v[2]) + " | recall " + ", ".join(rec))


if __name__ == "__main__":
    main(*sys.argv[1:])
