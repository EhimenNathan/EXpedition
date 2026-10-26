"""Region-majority kurgan class (LiDAR), nested + robust. The deployed decoder dual-emits both kurgan classes at almost
every peak (dual 0.05). Per site, q = confidence-weighted damaged share of the PRIMARY (higher-confidence) emissions;
if q ≥ θ the secondary intact emissions are dropped ("prune") or every intact emission becomes damaged ("relabel",
deduplicated), symmetric for q ≤ 1 − θ. Needs ≥ n peaks. Deployed blend + neighbourhood prior, thr 0.15.
  python scripts/li_regclass_oof.py
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main():
    from shapely.geometry import shape
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from robust_accept import site_counts, report
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    sites = sorted(P["ens"]); fold = {r: items[r]["fold"] for r in sites}; folds = sorted(set(fold.values()))
    pri = {"R": 200.0, "K": 3, "snb": 0.2}; D = {}
    for r in sites:
        b = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32); it = items[r]
        fs = decode_site_v2(b, it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05, comp_thr=0.5, prior=pri)
        for f in fs:
            c = shape(f["geometry"]).centroid; f["_k"] = (round(c.x, 1), round(c.y, 1))
        D[r] = fs
    print("decoded", len(sites), flush=True)

    def site_fc(st, r):
        mode, th, n = st; fs = D[r]
        if mode == "base":
            return [f for f in fs]
        kf = [f for f in fs if f["properties"]["class_name"] in K]
        best = {}
        for f in kf:
            if f["_k"] not in best or f["properties"]["confidence"] > best[f["_k"]]["properties"]["confidence"]:
                best[f["_k"]] = f
        prim = list(best.values())
        if len(prim) < n:
            return list(fs)
        w = np.array([f["properties"]["confidence"] for f in prim]); dm = np.array([f["properties"]["class_name"] == K[0] for f in prim])
        q = (w * dm).sum() / max(w.sum(), 1e-9)
        maj = K[0] if q >= th else (K[1] if q <= 1 - th else None)
        if maj is None:
            return list(fs)
        mino = K[1] if maj == K[0] else K[0]; out = [f for f in fs if f["properties"]["class_name"] not in K]
        for k_, f in best.items():
            pass
        if mode == "prune":         # drop minority emissions that are secondary at their location
            out += [f for f in kf if f["properties"]["class_name"] == maj or best[f["_k"]] is f]
        else:                       # relabel: every location → majority class (one emission, max confidence)
            for k_, f in best.items():
                g = dict(f); g["properties"] = dict(f["properties"], class_name=maj); out.append(g)
        return out
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    clean = lambda fs: {"type": "FeatureCollection", "features": [{k: v for k, v in f.items() if k != "_k"} for f in fs]}
    SS = [("base", 0, 0)] + [(m, th, n) for m in ("prune", "relabel") for th in (0.7, 0.8, 0.9) for n in (5, 15)]
    C = fold_counts(lambda st, f: clean([g for r in sites if fold[r] == f for g in site_fc(st, r)]), SS, folds, gt, CLS)
    for st in SS:
        tot = _add_all(C[st].values())
        print(f"{st}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  kt {tuple(int(v) for v in tot[K[1]])}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    CS = site_counts(lambda s, r: clean(site_fc(s, r)), (SS[0], os_), sites, gtr, CLS)
    report(f"LiDAR region-majority {os_}", CS[SS[0]], CS[os_])


if __name__ == "__main__":
    main()
