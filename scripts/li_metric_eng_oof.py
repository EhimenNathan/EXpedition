"""LiDAR metric engineering (nested over the site folds, copy-paste/openness OOF):
  A) per-class kurgan thresholds: decode at a low peak threshold, then keep damaged ≥ t_kp and intact ≥ t_kt
     (classes are scored independently, so each threshold is chosen on the other folds for its own class);
  B) regional cross-modal context (MIL-style), on regions labelled in LiDAR and satellite: where the satellite fusion
     has ≥ n kurgans in the region, LiDAR kurgans are emitted down to t_low instead of the deployed threshold.
"""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(oof="work/li/oof_augx.pkl", base_t="0.2", lows="0.1,0.15"):
    base_t = float(base_t); lows = [float(v) for v in lows.split(",")]
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    from koz3.score import score
    from ae_det_oof import f1
    from koz3.score import WEIGHTS
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = pickle.load(open(os.path.join(ROOT, oof), "rb"))
    fold = {r: items[r]["fold"] for r in P}; folds = sorted(set(fold.values()))
    gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if fold[r] == f])]) for f in folds}
    low = {r: decode_site_v2(P[r].astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=min(lows + [0.1]), dual=0.05) for r in P}
    # ---- A) per-class thresholds
    TH = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
    C = {}
    for t in TH:
        for f in folds:
            feats = [g for r in P if fold[r] == f for g in low[r] if g["properties"]["class_name"] not in K or g["properties"]["confidence"] >= t]
            _, res = score({"type": "FeatureCollection", "features": feats}, gt[f], CLS)
            for c, v in res.items():
                C[(c, t, f)] = (v["tp"], v["fp"], v["fn"])
    tot, chosen = {}, {}
    for c in CLS:
        for f in folds:
            t = 0.2 if c not in K else max(TH, key=lambda t: f1(*np.sum([C.get((c, t, g), (0, 0, 0)) for g in folds if g != f], 0)))
            chosen.setdefault(c, []).append(t); tot[c] = np.add(tot.get(c, (0, 0, 0)), C.get((c, t, f), (0, 0, 0)))
    base = {c: np.sum([C.get((c, 0.2, f), (0, 0, 0)) for f in folds], 0) for c in CLS}
    pres = [c for c in CLS if base[c][0] + base[c][2]]
    W = lambda tt: sum(WEIGHTS[c] * f1(*tt[c]) for c in pres) / sum(WEIGHTS[c] for c in pres)
    print(f"A) shared kurgan thr 0.2: {W(base):.4f}  | per-class nested: {W(tot):.4f}  chosen kp {chosen[K[0]]} kt {chosen[K[1]]}"
          f"  (kp F1 {f1(*base[K[0]]):.3f}→{f1(*tot[K[0]]):.3f}, kt {f1(*base[K[1]]):.3f}→{f1(*tot[K[1]]):.3f})")
    # ---- B) regional cross-modal context on LiDAR + satellite regions
    import geopandas as gpd, pandas as pd
    from sp_assemble_oof import models
    from sp_fusion_oof import fuse
    M = models(extra=["dfnodn", "balq"]); sp = fuse(list(M.values()), np.ones(len(M))); sp = sp[sp.score >= 0.25]
    Lall = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    both = sorted(set(Lall[Lall["mod"] == "Li"].region) & set(Lall[Lall["mod"] == "SpOr"].region) & set(P) & set(sp.region))
    nsp = sp[sp.cls.isin(K)].groupby("region").size()
    fb = sorted({fold[r] for r in both}); gtb = {f: to_gt_geojson(L[L.region.isin([r for r in both if fold[r] == f])]) for f in fb}

    def make(s, f):
        n, tl = s; feats = []
        for r in both:
            if fold[r] != f:
                continue
            t = tl if (n < 99 and nsp.get(r, 0) >= n) else base_t
            feats += [g for g in low[r] if g["properties"]["class_name"] not in K or g["properties"]["confidence"] >= t]
        return {"type": "FeatureCollection", "features": feats}
    S = [(99, base_t)] + [(n, tl) for n in (3, 10, 25) for tl in lows]
    CB = fold_counts(make, S, fb, gtb, CLS)
    from koz3.nested import weighted, _add_all
    for s in S:
        print(f"B) n≥{s[0]} t_low {s[1]}: pooled {weighted(_add_all(CB[s].values())):.4f}")
    h, ch, o, os_, _ = nested(CB)
    print(f"B) {len(both)} regions: no context {weighted(_add_all(CB[(99, base_t)].values())):.4f} | nested honest {h:.4f} chosen {list(ch.values())}")


if __name__ == "__main__":
    main(*sys.argv[1:])
