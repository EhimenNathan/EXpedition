"""object_poly ensemble: our CNN v1 OOF probability (0.1 m) + previous team's pixel-LightGBM LORO OOF probability
(per-method grid, reprojected onto ours). p = w·cnn + (1−w)·gbm → components → polygons; (w, thr, min area) chosen
nested over our 5 region-group folds."""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.environ.setdefault("GEO_INPUT", "rgb")


def main():
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling
    from koz3.geo_train import load_all, eval_maps, objects_from_mask, to_3857_fn, gt_for, FOLDS
    from koz3 import geo_data
    from koz3.nested import fold_counts, nested
    from shapely.ops import transform as stx
    items = {it["name"]: it for it in load_all()}
    P = pickle.load(open(os.path.join(ROOT, "work/geo/oof_probs.pkl"), "rb"))
    units = {f"{u['region']}|{u['method']}|{u['key']}": u for u in geo_data.discover(os.path.join(ROOT, "kaggle_import/code/data_geo"))}
    gbm_dir = os.path.join(ROOT, "kaggle_import/code/work/geo/objects_v3/oof")
    maps = []
    for name, (prob, _) in P.items():
        it = items[name]; m = it["meta"]
        key = "" if (m["method"] == "MagGr" and m["key"] in ("серый", "цвет")) else m["key"]
        uid = f"{m['region']}|{m['method']}|{key}"
        f = os.path.join(gbm_dir, uid.replace("|", "__") + ".npz")
        if not os.path.exists(f) or uid not in units:
            print("no gbm map for", uid); continue
        z = np.load(f); gp = z["prob"].astype(np.float32)
        gsd = {"MagGr": 0.2, "EtGr": 0.1, "TerFl": 0.3}[m["method"]]
        u = units[uid]
        with rasterio.open(u["path"]) as r:
            fct = max(1, int(round(gsd / abs(r.transform.a)))); H, W = max(1, r.height // fct), max(1, r.width // fct)
            Tg = r.transform * r.transform.scale(r.width / W, r.height / H); crs = r.crs
        dst = np.zeros(it["valid"].shape, np.float32)
        reproject(gp, dst, src_transform=Tg, src_crs=crs, dst_transform=Affine(*m["transform"]), dst_crs=crs,
                  resampling=Resampling.bilinear)
        maps.append((m["region"], it["fold"], prob[1].astype(np.float32), dst, it["valid"], m))
    print(len(maps), "maps with both probabilities")
    regions = sorted({x[0] for x in maps}); folds = sorted({x[1] for x in maps})
    gt_by_fold = {f: gt_for([x[0] for x in maps if x[1] == f]) for f in folds}
    cache = {}

    def make(s, f):
        w, thr, mn = s
        feats = []
        for i, (reg, fo, pc, pg, v, m) in enumerate(maps):
            if fo != f:
                continue
            k = (i, w, thr, mn)
            if k not in cache:
                fn = to_3857_fn(m); out = []
                for poly, sc in objects_from_mask(w * pc + (1 - w) * pg, v, thr, int(mn / 0.01)):
                    out.append({"type": "Feature", "geometry": stx(lambda x, y, z=None: fn(x, y), poly).__geo_interface__,
                                "properties": {"class_name": "object_poly", "confidence": sc}})
                cache[k] = out
            feats += cache[k]
        return {"type": "FeatureCollection", "features": feats}
    S = [(w, t, mn) for w in (0.0, 0.3, 0.5, 0.7, 1.0) for t in (0.3, 0.4, 0.5) for mn in (0.5, 1.5)]
    C = fold_counts(make, S, folds, gt_by_fold, ["object_poly"])
    for w in (0.0, 0.5, 1.0):
        sub = {k: v for k, v in C.items() if k[0] == w}
        h, ch, o, os_, _ = nested(sub); print(f"w={w}: honest {h:.4f} oracle {o:.4f} @ {os_}")
    h, ch, o, os_, tot = nested(C)
    print(f"ALL: honest {h:.4f} oracle {o:.4f} @ {os_}  chosen {ch}  {tot}")


if __name__ == "__main__":
    main()
