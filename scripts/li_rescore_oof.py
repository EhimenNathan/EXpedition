"""LiDAR kurgan morphometric rescoring (the "geomorphometric constraints / LOF on verified kurgan morphology" idea,
done supervised and nested): every kurgan peak of the deployed LiDAR ensemble OOF (decode at a low threshold, one
emission per peak) gets physical features — local relief at 2/5/10/25 m (centre, rings 1–12 m, ring std = asymmetry),
slope, curvature, summit dent, ring ditch, height, the model's heat / mask probabilities, neighbour support. A LightGBM
trained on the OTHER folds only predicts "this peak is the best match of a kurgan label"; kurgan peaks are then
re-emitted with blended confidence and thresholds chosen on the other folds. Non-kurgan classes unchanged.

  python scripts/li_rescore_oof.py [work/li/oof_ens.pkl]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from li_kurgan_cls import feats_at  # noqa: E402

K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(oof="work/li/oof_ens.pkl"):
    import lightgbm as lgb
    from pyproj import Transformer
    from shapely import STRtree
    from shapely.geometry import Point, shape
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = pickle.load(open(os.path.join(ROOT, oof), "rb"))
    cache = os.path.join(ROOT, "work/li/rescore_cands.parquet")
    if os.path.exists(cache) and os.environ.get("REBUILD") != "1":
        D = pd.read_parquet(cache)
    else:
        rows = []
        for r, p in P.items():
            it = items[r]; meta = dict(it["meta"], region=r); p = p.astype(np.float32); x = np.asarray(it["x"], np.float32)
            feats = decode_site_v2(p, it["valid"], meta, thr=0.05, dual=1.1)
            tr = Transformer.from_crs(3857, meta["crs"], always_xy=True)
            lab = L[(L.region == r) & L.class_name.isin(K)].to_crs(meta["crs"])
            geoms = list(lab.geometry); tree = STRtree(geoms) if geoms else None
            best = {}
            cands = []
            for f in feats:
                c = f["properties"]["class_name"]
                if c not in K:
                    continue
                cx, cy = shape(f["geometry"]).centroid.coords[0]
                X, Y = tr.transform(cx, cy)
                col, row = int((X - meta["x0"]) / meta["res"]), int((meta["ytop"] - Y) / meta["res"])
                if not (0 <= row < x.shape[1] and 0 <= col < x.shape[2]):
                    continue
                d = feats_at(x, p, row, col)
                d.update(region=r, fold=it["fold"], cls=c, x=cx, y=cy, conf=f["properties"]["confidence"], lab=0)
                i = len(cands); cands.append(d)
                if tree is not None:
                    pt = Point(X, Y)
                    for j in tree.query(pt):
                        if geoms[j].contains(pt) and d["peak"] > best.get(j, (-1, -1))[0]:
                            best[j] = (d["peak"], i)
            for _, i in best.values():
                cands[i]["lab"] = 1
            if cands:
                xy = np.array([[c["x"], c["y"]] for c in cands]); t = cKDTree(xy); pk = np.array([c["peak"] for c in cands])
                for R in (10, 25, 60):
                    nb = t.query_ball_point(xy, R)
                    for i, n in enumerate(nb):
                        cands[i][f"nb{R}"] = len(n) - 1
                        cands[i][f"nbpk{R}"] = float(np.mean(pk[[j for j in n if j != i]])) if len(n) > 1 else 0.0
            rows += cands
        D = pd.DataFrame(rows); D.to_parquet(cache)
    print(len(D), "kurgan peaks;", int(D.lab.sum()), "best matches of a kurgan label")
    feats = [c for c in D.columns if c not in ("region", "fold", "cls", "x", "y", "conf", "lab")]
    from rescore_models import oof_predict
    kinds = os.environ.get("LI_RESCORE_MODELS", "lgbm").split(",")       # e.g. lgbm,catboost,ngboost,avg,catboost+anom
    for kd in kinds:
        D["p_" + kd.replace("+", "_")] = oof_predict(D, feats, kd)
    D["p"] = D["p_" + kinds[0].replace("+", "_")]
    from sklearn.metrics import roc_auc_score
    print(f"AUC peak heat {roc_auc_score(D.lab, D.peak):.3f}  conf {roc_auc_score(D.lab, D.conf):.3f}  → " + " ".join(f"{kd} {roc_auc_score(D.lab, D['p_' + kd.replace('+', '_')]):.3f}" for kd in kinds))
    # nested metric: non-kurgan features from the deployed decode; kurgan peaks re-emitted (deployed dual rule)
    folds = sorted(D.fold.unique())
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if items[r]["fold"] == f])]) for f in folds}
    others = {r: [g for g in decode_site_v2(P[r].astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=0.15, dual=0.05)
                  if g["properties"]["class_name"] not in K] for r in P}
    deployed = {r: [g for g in decode_site_v2(P[r].astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=0.15, dual=0.05)
                    if g["properties"]["class_name"] in K] for r in P}
    kk = lambda yy: 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(yy / 6378137.0)) - np.pi / 2)))
    for kd in kinds:
        D["b_" + kd.replace("+", "_")] = np.sqrt(D.peak * D["p_" + kd.replace("+", "_")])
    srcs = ("peak",) + tuple(f"{pre}_{kd.replace('+', '_')}" for kd in kinds for pre in ("p", "b"))

    def make(s, f):
        src, t = s
        out = [g for r in P if items[r]["fold"] == f for g in others[r]]
        if src == "deployed":
            return {"type": "FeatureCollection", "features": out + [g for r in P if items[r]["fold"] == f for g in deployed[r]]}
        G = D[(D.fold == f) & (D[src] >= t)]
        for row in G.itertuples():
            X, Y = row.x, row.y                     # stored in EPSG:3857
            pdam = row.p_ratio
            emit = [("kurgany_povrezhdennye", pdam), ("kurgany_tselye", 1 - pdam)]; emit.sort(key=lambda e: -e[1])
            emit = [emit[0]] + [e for e in emit[1:] if e[1] >= 0.05]
            for c, pc in emit:
                out.append({"type": "Feature", "geometry": Point(X, Y).buffer(kk(Y), 12).__geo_interface__,
                            "properties": {"class_name": c, "confidence": float(getattr(row, src) * pc)}})
        return {"type": "FeatureCollection", "features": out}
    S = [("deployed", 0)] + [(s_, t) for s_ in srcs for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)]
    C = fold_counts(make, S, folds, gt, CLS)
    for src in ("deployed",) + srcs:
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[0] == src})
        print(f"{src:9s} honest {h:.4f} (oracle {o:.4f} @ {os_[1]}) " + " ".join(f"{c[:8]} {2 * v[0] / max(2 * v[0] + v[2] + 0.5 * v[1], 1):.3f}" for c, v in tot.items() if v[0] + v[2]))


if __name__ == "__main__":
    main(*sys.argv[1:])
