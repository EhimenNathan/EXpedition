"""Honest per-fold LiDAR ensembles: for each fold, average the probability maps of the seed-0 and seed-1 models (both
trained without that fold) on its held-out sites; nested selection of the decoding settings (decode_site_v2).
Compares against the single-model OOF (seed 0)."""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def main(tags=("", "_s1")):
    from koz3 import hm
    from koz3.li_train import labels, site_items, load_model, decode_site_v2, CLS, N_HM, N_MK
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    L = labels(); items = site_items(L)
    out = os.path.join(ROOT, "work/li/oof_ens.pkl")
    if os.path.exists(out):
        probs = pickle.load(open(out, "rb"))
    else:
        probs = {}
        for fold in sorted({it["fold"] for it in items if it["fold"] >= 0}):
            ms = [load_model(os.path.join(ROOT, f"work/li/li{t}_f{fold}.pt")) for t in tags]
            for it in [i for i in items if i["fold"] == fold]:
                x = np.asarray(it["x"], np.float32)
                probs[it["region"]] = np.mean([hm.predict_tiled(m, x, N_HM + N_MK) for m in ms], 0).astype(np.float16)
            print("ensemble oof fold", fold, flush=True)
        pickle.dump(probs, open(out, "wb"))
    single = pickle.load(open(os.path.join(ROOT, "work/li/oof.pkl"), "rb"))
    byr = {it["region"]: it for it in items}
    fold = {r: byr[r]["fold"] for r in probs}; folds = sorted(set(fold.values()))
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in probs if fold[r] == f])]) for f in folds}
    for name, P in (("single seed0", single), ("ensemble s0+s1", probs)):
        cache = {}

        def make(s, f):
            t, comp = s; feats = []
            for r, p in P.items():
                if fold[r] != f:
                    continue
                k = (r, t, comp)
                if k not in cache:
                    it = byr[r]
                    cache[k] = decode_site_v2(p.astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=t, dual=0.05, comp_thr=comp)
                feats += cache[k]
            return {"type": "FeatureCollection", "features": feats}
        S = [(t, c) for t in (0.05, 0.08, 0.1, 0.15) for c in (0.4, 0.5)]
        C = fold_counts(make, S, folds, gt, CLS)
        h, ch, o, os_, tot = nested(C)
        print(f"{name:16s} honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {ch}")
        for c, (tp, fp, fn) in tot.items():
            if tp + fn:
                print(f"    {c:24s} F1 {2 * tp / max(2 * tp + fn + 0.5 * fp, 1):.3f}")


if __name__ == "__main__":
    main()
