"""Per-site heat recalibration (Vesuvius ink detection: per-fragment percentile normalisation / adaptive thresholds),
nested + robust on the deployed LiDAR decoder (4-model blend, neighbourhood prior, thr 0.15, dual 0.05).
Whole sites are "blind" at unseen locations (uniformly weak heat, same model 93–100 % recall when the site is in
training) → scale the kurgan heat channel of each site by f = clip(c / m, lo, cap), where m = site statistic:
  "topk": mean of the site's top-10 raw kurgan peaks;  "pct": 99.95th percentile of the valid kurgan heat.
  python scripts/li_site_calib_oof.py
"""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    import koz3.hm as hm
    from koz3.li_train import labels, site_items, decode_site_v2, CLS, N_HM
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from robust_accept import site_counts, report
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    sites = sorted(P["ens"]); fold = {r: items[r]["fold"] for r in sites}; folds = sorted(set(fold.values()))
    B, stat = {}, {}
    for r in sites:
        b = ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32); B[r] = b
        pk = sorted([q[3] for q in hm.decode(b[:N_HM], items[r]["valid"], thr=0.02, nms=11) if q[0] == 0], reverse=True)
        v = b[0][np.asarray(items[r]["valid"], bool)] if items[r]["valid"] is not None else b[0].ravel()
        stat[r] = {"topk": float(np.mean(pk[:10])) if pk else 0.0, "pct": float(np.percentile(v, 99.95)) if v.size else 0.0}
    for r in sites:
        print(f"  {r:40s} topk {stat[r]['topk']:.3f}  pct {stat[r]['pct']:.3f}")
    pri = {"R": 200.0, "K": 3, "snb": 0.2}; cache = {}

    def site_fc(st, r):
        key = (st, r)
        if key not in cache:
            b = B[r]
            if st[0] != "base":
                mode, c, lo, cap = st
                f = float(np.clip(c / max(stat[r][mode], 1e-3), lo, cap))
                if f != 1.0:
                    b = b.copy(); b[0] = np.minimum(b[0] * f, 1.0)
            it = items[r]
            cache[key] = decode_site_v2(b, it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05, comp_thr=0.5, prior=pri)
        return cache[key]
    SS = [("base",)] + [(m, c, lo, cap) for m, c, lo, cap in itertools.product(("topk", "pct"), (0.3, 0.5, 0.7), (1.0, 0.75), (1.5, 2.5))]
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in sites if fold[r] == f])]) for f in folds}
    C = fold_counts(lambda st, f: {"type": "FeatureCollection", "features": [g for r in sites if fold[r] == f for g in site_fc(st, r)]}, SS, folds, gt, CLS)
    for st in SS:
        print(f"{st}: pooled {weighted(_add_all(C[st].values())):.4f}", flush=True)
    h, ch, o, os_, _ = nested(C)
    print(f"deployed {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    CS = site_counts(lambda s, r: {"type": "FeatureCollection", "features": site_fc(s, r)}, (SS[0], os_), sites, gtr, CLS)
    report(f"LiDAR site recalibration {os_}", CS[SS[0]], CS[os_])


if __name__ == "__main__":
    main()
