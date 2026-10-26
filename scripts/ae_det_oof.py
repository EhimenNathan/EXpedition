"""Ae nested study: Ae-trained detector (koz3/ae_det_train.py OOF) vs the deployed Ae expert (SpOr D-FINE, damaged
kurgans ≥ 0.7), on Ae labels, folds = Ae's own site folds. Classes are scored independently by the official metric, so
each class's threshold is chosen per held-out fold on the other folds (nested); "off" = emit nothing for that class.

  python scripts/ae_det_oof.py [tag ...]      (default tag ae_fcos)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_fusion_oof import fc  # noqa: E402

TH = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)


MOD = os.environ.get("DET_MOD", "Ae")          # DET_MOD=Or LORO=1 → Or, leave-one-region-out


def ae_folds():
    from koz3.sp_train import load_chips
    return {c["region"]: c["fold"] for c in load_chips(MOD)}


def per_class_counts(D, gt, folds, RF):
    """→ {cls: {thr: {fold: (tp, fp, fn)}}}"""
    from koz3.score import score
    out = {}
    for t in TH:
        for f in folds:
            _, res = score(fc(D[(D.score >= t) & (D.region.map(RF) == f)]), gt[f])
            for c, v in res.items():
                out.setdefault(c, {}).setdefault(t, {})[f] = (v["tp"], v["fp"], v["fn"])
    return out


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fn + 0.5 * fp, 1e-9) if tp + fn else 0.0


def nested_per_class(C, folds):
    from koz3.score import WEIGHTS
    tot, chosen = {}, {}
    for c, byt in C.items():
        for f in folds:
            others = {t: np.sum([v[g] for g in folds if g != f], 0) for t, v in byt.items()}
            t = max(others, key=lambda t: f1(*others[t]))
            chosen.setdefault(c, []).append(t)
            tot[c] = np.add(tot.get(c, (0, 0, 0)), byt[t][f])
    present = [c for c, v in tot.items() if v[0] + v[2]]
    w = sum(WEIGHTS[c] for c in present)
    return sum(WEIGHTS[c] * f1(*tot[c]) for c in present) / w, {c: (round(f1(*tot[c]), 3), tuple(int(x) for x in tot[c]), chosen[c]) for c in present}


def main(tags):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    RF = ae_folds(); folds = sorted(set(RF.values()))
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == MOD) & L.region.isin(RF)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    print(len(RF), "Ae regions, folds", {f: sum(v == f for v in RF.values()) for f in folds}, "GT", len(L))
    cp = os.path.join(ROOT, f"work/{MOD.lower()}/oof_spdet_cands.parquet")
    cur = pd.read_parquet(cp) if os.path.exists(cp) else pd.DataFrame(columns=["region", "cls", "score", "x", "y"])
    cur = cur[(cur.cls == "kurgany_povrezhdennye") & (cur.score >= 0.7)] if MOD == "Ae" else cur[(cur.cls == "arkhitektury") & (cur.score >= 0.5)]
    tot = {}
    for f in folds:
        _, res = score(fc(cur[cur.region.map(RF) == f]), gt[f])
        for c, v in res.items():
            tot[c] = np.add(tot.get(c, (0, 0, 0)), (v["tp"], v["fp"], v["fn"]))
    from koz3.score import WEIGHTS
    present = [c for c, v in tot.items() if v[0] + v[2]]
    print(f"{'deployed (SpOr D-FINE)':32s} {sum(WEIGHTS[c] * f1(*tot[c]) for c in present) / sum(WEIGHTS[c] for c in present):.4f}",
          {c: round(f1(*tot[c]), 3) for c in present})
    for tag in tags:
        D = pd.read_parquet(os.path.join(ROOT, f"work/ae/oof_{tag}_cands.parquet"))
        h, per = nested_per_class(per_class_counts(D, gt, folds, RF), folds)
        print(f"{tag:32s} honest {h:.4f}")
        for c, v in per.items():
            print(f"    {c:24s} F1 {v[0]}  tp/fp/fn {v[1]}  thr per fold {v[2]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ae_fcos"])
