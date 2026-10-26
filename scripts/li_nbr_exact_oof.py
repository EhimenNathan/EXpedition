"""LiDAR neighbourhood prior inside the EXACT deployed decoding (decode_site_v2): each heatmap peak's height s is
multiplied by w = 0.5 + 0.5·clip(n/K, 0, 2), n = other kurgan peaks (height ≥ s_nb) within R metres, before the deployed
peak threshold; v12 blend (pair avg + aug + augx), dual 0.05, comp 0.5. Nested over (prior params, thr)."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    import koz3.hm as hm
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    import li_compare_oof as lc
    P = lc.load("ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst")
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    fold = {r: items[r]["fold"] for r in P}; folds = sorted(set(fold.values()))
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if fold[r] == f])]) for f in folds}
    orig = hm.decode
    state = {}

    def decode_prior(h, valid=None, thr=0.05, nms=9):
        R, Kc, snb = state["p"]
        pk = orig(h, valid, min(thr, 0.05), nms)
        if R == 0 or not pk:
            return [q for q in pk if q[3] > thr]
        k0 = [q for q in pk if q[0] == 0]; strong = np.array([(q[1], q[2]) for q in k0 if q[3] >= snb]).reshape(-1, 2)
        t = cKDTree(strong) if len(strong) else None; out = []
        for c, x, y, s in pk:
            if c == 0 and t is not None:
                n = len(t.query_ball_point((x, y), R / 0.5)) - (1 if s >= snb else 0)
                s = s * (0.5 + 0.5 * np.clip(n / Kc, 0, 2))
            elif c == 0:
                s = s * 0.5
            if s > thr:
                out.append((c, x, y, float(s)))
        return out
    hm.decode = decode_prior
    cache = {}

    def make(s, f):
        prm, t = s; state["p"] = prm; feats = []
        for r in P:
            if fold[r] != f:
                continue
            key = (r, prm, t)
            if key not in cache:
                it = items[r]; cache[key] = decode_site_v2(P[r].astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=t, dual=0.05, comp_thr=0.5)
            feats += cache[key]
        return {"type": "FeatureCollection", "features": feats}
    SS = [((0, 0, 0), t) for t in (0.1, 0.15, 0.2)] + [((R, Kc, snb), t) for R, Kc in ((100, 3), (200, 3)) for snb in (0.1, 0.2) for t in (0.08, 0.1, 0.15)]
    C = fold_counts(make, SS, folds, gt, CLS)
    for s in SS:
        print(f"prior {s[0]} thr {s[1]}: pooled {weighted(_add_all(C[s].values())):.4f}", flush=True)
    base = {k: v for k, v in C.items() if k[0] == (0, 0, 0)}
    h0, ch0, *_ = nested(base); h, ch, o, os_, _ = nested(C)
    print(f"deployed decoding nested {h0:.4f} (chosen {list(ch0.values())}) | with prior nested {h:.4f} (oracle {o:.4f}) chosen {list(ch.values())}")


if __name__ == "__main__":
    main()
