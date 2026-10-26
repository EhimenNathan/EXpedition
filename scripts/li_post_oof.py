"""LiDAR post-processing study on saved OOF probabilities (nested over folds): lower peak thresholds, and
fortifications / hillforts from mask components (representative point of each component) instead of peaks."""
import os, pickle, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3.li_train import labels, site_items, decode_site, CLS, N_HM  # noqa: E402


def comp_feats(prob_mask, valid, meta, cls, thr, min_px, r_out=1.0):
    import cv2
    from pyproj import Transformer
    from shapely.geometry import Polygon, Point
    m = ((prob_mask > thr) & valid).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True); out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < min_px:
            continue
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        if len(c) < 4:
            continue
        p = Polygon(c[:, 0, :] + 0.5).buffer(0)
        if p.is_empty:
            continue
        p = max(p.geoms, key=lambda g: g.area) if p.geom_type != "Polygon" else p
        rp = p.representative_point()
        mx, my = tr.transform(meta["x0"] + rp.x * meta["res"], meta["ytop"] - rp.y * meta["res"])
        kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2)))
        out.append({"type": "Feature", "geometry": Point(mx, my).buffer(r_out * kk, 12).__geo_interface__,
                    "properties": {"class_name": cls, "confidence": float(prob_mask[lab == k].mean()), "region_name": meta["region"], "markup_type": "li"}})
    return out


def main():
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    probs = pickle.load(open(os.path.join(ROOT, "work/li/oof.pkl"), "rb"))
    fold = {r: items[r]["fold"] for r in probs}; folds = sorted(set(fold.values()))
    gt_by_fold = {f: to_gt_geojson(L[L.region.isin([r for r in probs if fold[r] == f])]) for f in folds}
    peak_cache, comp_cache = {}, {}
    def peaks(r, t, dual):
        if (r, t, dual) not in peak_cache:
            it = items[r]
            peak_cache[(r, t, dual)] = decode_site(probs[r].astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=(t, t, t), dual=dual)
        return peak_cache[(r, t, dual)]
    def comps(r, ci, thr, min_px):
        key = (r, ci, thr, min_px)
        if key not in comp_cache:
            it = items[r]
            comp_cache[key] = comp_feats(probs[r][N_HM + ci].astype(np.float32), it["valid"], dict(it["meta"], region=r), CLS[ci], thr, min_px)
        return comp_cache[key]
    settings = []
    for t in (0.02, 0.03, 0.05, 0.1):
        settings.append((t, "peaks", None, None))
        for fthr in (0.3, 0.5):
            for fmin in (8, 30):
                settings.append((t, "fort_comp", fthr, fmin))
                settings.append((t, "fort+gor_comp", fthr, fmin))
    def make(s, f):
        t, mode, thr, mn = s
        feats = []
        for r in probs:
            if fold[r] != f:
                continue
            pk = peaks(r, t, 0.3)
            if mode == "peaks":
                feats += pk; continue
            drop = {"fortifikatsii"} if mode == "fort_comp" else {"fortifikatsii", "gorodishcha"}
            feats += [x for x in pk if x["properties"]["class_name"] not in drop]
            feats += comps(r, CLS.index("fortifikatsii"), thr, mn)
            if mode == "fort+gor_comp":
                feats += comps(r, CLS.index("gorodishcha"), thr, mn * 20)
        return {"type": "FeatureCollection", "features": feats}
    C = fold_counts(make, settings, folds, gt_by_fold, CLS)
    sub = {k: v for k, v in C.items() if k[1] == "peaks"}
    h0, ch0, o0, _, _ = nested(sub)
    print(f"peaks only (lower grid): honest {h0:.4f} oracle {o0:.4f} chosen {ch0}")
    h, ch, o, os_, tot = nested(C)
    print(f"all settings: honest {h:.4f} oracle {o:.4f} @ {os_}\nchosen {ch}")
    for c, (tp, fp, fn) in tot.items():
        if tp + fn:
            print(f"   {c:24s} F1 {2*tp/max(2*tp+fn+0.5*fp,1):.3f} tp {tp} fp {fp} fn {fn}")


if __name__ == "__main__":
    main()
