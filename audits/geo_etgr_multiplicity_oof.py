"""Prospective test of preserving repeated EtGr annotations in official matching.

The training set contains many match-compatible object_poly labels across separate
resistivity profiles.  Test fixed replication of each independently inferred EtGr
polygon.  Every setting is scored on geographic folds with nested selection.
"""
import os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("GEO_INPUT", "rgb")


def main():
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling
    from shapely.geometry import Point
    from shapely.ops import transform as stx, polylabel
    from koz3.geo_train import load_all, objects_from_mask, to_3857_fn, gt_for
    from koz3 import geo_data
    from koz3.nested import fold_counts, nested, weighted, _add_all

    items = {it["name"]: it for it in load_all()}
    P = pickle.load(open(os.path.join(ROOT, "work/geo/oof_probs.pkl"), "rb"))
    units = {f"{u['region']}|{u['method']}|{u['key']}": u for u in geo_data.discover(os.path.join(ROOT, "kaggle_import/code/data_geo"))}
    gbm_dir = os.path.join(ROOT, "kaggle_import/code/work/geo/objects_v3/oof")
    maps = []
    for name, (prob, _) in P.items():
        it = items[name]; m = it["meta"]
        key = "" if (m["method"] == "MagGr" and m["key"] in ("серый", "цвет")) else m["key"]
        uid = f"{m['region']}|{m['method']}|{key}"
        path = os.path.join(gbm_dir, uid.replace("|", "__") + ".npz")
        if not os.path.exists(path) or uid not in units: continue
        gp = np.load(path)["prob"].astype(np.float32)
        gsd = {"MagGr": .2, "EtGr": .1, "TerFl": .3}[m["method"]]
        u = units[uid]
        with rasterio.open(u["path"]) as r:
            fact = max(1, int(round(gsd / abs(r.transform.a))))
            H, W = max(1, r.height // fact), max(1, r.width // fact)
            Tg = r.transform * r.transform.scale(r.width / W, r.height / H); crs = r.crs
        dst = np.zeros(it["valid"].shape, np.float32)
        reproject(gp, dst, src_transform=Tg, src_crs=crs,
                  dst_transform=Affine(*m["transform"]), dst_crs=crs,
                  resampling=Resampling.bilinear)
        maps.append((m["region"], it["fold"], prob[1].astype(np.float32), dst, it["valid"], m, u["kind"]))

    folds = sorted({x[1] for x in maps})
    gt_by_fold = {f: gt_for(sorted({x[0] for x in maps if x[1] == f})) for f in folds}
    cache = {}

    from koz3.dedupe import dedupe_region

    def make(s, fold):
        copies, prof_only, dedupe_mode = s
        out = []
        for i, (reg, fo, pc, pg, valid, m, kind) in enumerate(maps):
            if fo != fold: continue
            method = m["method"]
            thr = .75 if method == "MagGr" else .3
            key = (i, thr)
            if key not in cache:
                fn = to_3857_fn(m); fs = []
                for poly, score in objects_from_mask(.5*pc + .5*pg, valid, thr, 50):
                    q = polylabel(poly, tolerance=.5)
                    anchor = Point(q.x, q.y).buffer(max(.5, min(10., .8*poly.exterior.distance(q))), 12)
                    g = stx(lambda x,y,z=None: fn(x,y), anchor).__geo_interface__
                    fs.append({"type":"Feature", "geometry":g,
                               "properties":{"class_name":"object_poly", "confidence":score}})
                cache[key] = fs
            ncopy = copies if method == "EtGr" and (not prof_only or kind == "profile") else 1
            for rank in range(ncopy):
                for ft in cache[key]:
                    z = {"type":"Feature", "geometry":ft["geometry"], "properties":dict(ft["properties"])}
                    z["properties"]["confidence"] *= (1 - .001*rank)
                    z["properties"].update(region_name=reg, markup_type=method.lower(),
                                           image=str(m.get("key", "")))
                    out.append(z)
        if dedupe_mode == "current":
            out = dedupe_region(out, 3.0)
        elif dedupe_mode == "keep_etgr":
            et = [f for f in out if f["properties"]["markup_type"] == "etgr"]
            other = [f for f in out if f["properties"]["markup_type"] != "etgr"]
            out = et + dedupe_region(other, 3.0)
        return {"type":"FeatureCollection", "features":out}

    settings = [(1, False, mode) for mode in ("current", "keep_etgr", "none")]
    settings += [(n, p, "none") for p in (False, True) for n in (2,3,4,6,10)]
    counts = fold_counts(make, settings, folds, gt_by_fold, ["object_poly"])
    base = weighted(_add_all(counts[(1,False,"current")].values()))
    for s in settings:
        z = _add_all(counts[s].values())
        print(s, f"pooled={weighted(z):.6f} delta={weighted(z)-base:+.6f}", tuple(int(v) for v in z["object_poly"]))
    print("per-fold current -> keep_etgr")
    for f in folds:
        a = counts[(1,False,"current")][f]; b = counts[(1,False,"keep_etgr")][f]
        print(f, f"{weighted(a):.6f}->{weighted(b):.6f}",
              tuple(int(v) for v in a["object_poly"]), tuple(int(v) for v in b["object_poly"]))
    h, chosen, oracle, os_, total = nested(counts)
    print("nested", h, "oracle", oracle, os_, "chosen", chosen, "total", total)


if __name__ == "__main__": main()
