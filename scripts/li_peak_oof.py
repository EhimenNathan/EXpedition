"""LiDAR peak post-processing (nested over the 5 site folds, copy-paste/openness OOF): heatmap NMS window and
sub-pixel peak refinement (heat-weighted centroid of the 5×5 neighbourhood)."""
import os, sys, pickle
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)


def main():
    import koz3.hm as hm
    import koz3.li_train as lt
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    L = lt.labels(); items = {it["region"]: it for it in lt.site_items(L)}
    P = pickle.load(open(os.path.join(ROOT, "work/li/oof_augx.pkl"), "rb"))
    fold = {r: items[r]["fold"] for r in P}; folds = sorted(set(fold.values()))
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if fold[r] == f])]) for f in folds}
    orig_decode = hm.decode

    def decode_sub(h, valid=None, thr=0.05, nms=9):
        out = []
        for c, x, y, s in orig_decode(h, valid, thr, nms):
            yi, xi = int(y), int(x)
            y0, y1, x0, x1 = max(0, yi - 2), min(h.shape[1], yi + 3), max(0, xi - 2), min(h.shape[2], xi + 3)
            w = h[c, y0:y1, x0:x1].astype(np.float64); yy, xx = np.mgrid[y0:y1, x0:x1]
            out.append((c, float((w * (xx + 0.5)).sum() / w.sum()), float((w * (yy + 0.5)).sum() / w.sum()), s))
        return out
    cache = {}

    def make(s, f):
        nms, sub = s; feats = []
        hm.decode = decode_sub if sub else orig_decode
        orig_ds = lt.decode_site
        lt.decode_site = lambda *a, **k: orig_ds(*a, **dict(k, nms=nms))
        try:
            for r, p in P.items():
                if fold[r] != f:
                    continue
                key = (r, nms, sub)
                if key not in cache:
                    it = items[r]; cache[key] = lt.decode_site_v2(p.astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=0.2, dual=0.05)
                feats += cache[key]
        finally:
            hm.decode = orig_decode; lt.decode_site = orig_ds
        return {"type": "FeatureCollection", "features": feats}
    S = [(n, s) for n in (7, 9, 11, 15, 19) for s in (False, True)]
    C = fold_counts(make, S, folds, gt, lt.CLS)
    from koz3.nested import weighted, _add_all
    for s in S:
        print("pooled", s, round(weighted(_add_all(C[s].values())), 4))
    h, ch, o, os_, tot = nested(C)
    print(f"deployed (11, False) {weighted(_add_all(C[(11, False)].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {ch}")


if __name__ == "__main__":
    main()
