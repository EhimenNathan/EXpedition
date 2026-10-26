"""Nested geographic OOF audit of class-specific thresholds for v9 satellite fusion.

Selection for each held-out fold uses the other three folds only. The v9
hillfort scene gate stays fixed. This is a calibration audit, not a deployment.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.labels import to_gt_geojson
from koz3.score import score, WEIGHTS
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc

TS = (.10, .15, .20, .25, .30, .35, .40, .50)
CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii")


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + .5 * fp + fn, 1e-9)


def main():
    M = models(extra=("dfnodn",))
    F = fuse(list(M.values()), np.ones(len(M)))
    RF = region_folds()
    folds = sorted(set(RF.values()))
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    L = L[(L["mod"] == "SpOr") & L.region.isin(RF)]
    gt = {k: to_gt_geojson(L[L.region.map(RF) == k]) for k in folds}
    S = pd.read_parquet(ROOT / "work/spor/oof_scene_cls.parquet")
    gate = S[S.p >= .1]
    G = pd.DataFrame({"region": gate.region, "cls": "gorodishcha", "score": gate.p,
                      "x": gate.x, "y": gate.y3857})
    # Existing detector hillforts survive only at gate-positive sites.
    F = F[(F.cls != "gorodishcha") | F.region.isin(set(gate.region))]
    counts = {}
    for t in TS:
        for fold in folds:
            selected = F[(F.score >= t) & (F.region.map(RF) == fold)]
            chosen = pd.concat([selected, G[G.region.map(RF) == fold]], ignore_index=True)
            _, details = score(fc(chosen), gt[fold])
            for cls in CLASSES:
                d = details[cls]
                counts[(fold, cls, t)] = (d["tp"], d["fp"], d["fn"])
    def aggregate(selections):
        pooled = {c: np.sum([counts[(f, c, selections[(f, c)])] for f in folds], axis=0)
                  for c in CLASSES}
        w = sum(WEIGHTS[c] for c in CLASSES)
        s = sum(WEIGHTS[c] * f1(*pooled[c]) for c in CLASSES) / w
        return s, {c: tuple(map(int, pooled[c])) for c in CLASSES}
    fixed = {(f, c): .25 for f in folds for c in CLASSES}
    base, base_counts = aggregate(fixed)
    preset = {"kurgany_povrezhdennye": .30, "kurgany_tselye": .35,
              "gorodishcha": .35, "arkhitektury": .10, "fortifikatsii": .25}
    calibrated, cal_counts = aggregate({(f, c): preset[c] for f in folds for c in CLASSES})
    selected = {}
    for f in folds:
        for c in CLASSES:
            other = {t: np.sum([counts[(g, c, t)] for g in folds if g != f], axis=0)
                     for t in TS}
            selected[(f, c)] = max(TS, key=lambda t: (f1(*other[t]), -t))
    nested, nested_counts = aggregate(selected)
    print("folds", folds, "GT", len(L), "fused candidates", len(F))
    print("fixed t=.25", round(base, 6), base_counts)
    print("fixed class preset", round(calibrated, 6), cal_counts)
    print("nested class thresholds", round(nested, 6), nested_counts)
    for f in folds:
        fixed_fold = sum(WEIGHTS[c] * f1(*counts[(f, c, .25)]) for c in CLASSES) / sum(WEIGHTS[c] for c in CLASSES)
        nested_fold = sum(WEIGHTS[c] * f1(*counts[(f, c, selected[(f, c)])]) for c in CLASSES) / sum(WEIGHTS[c] for c in CLASSES)
        print("fold", f, "fixed", round(fixed_fold, 5), "nested", round(nested_fold, 5))
    for c in CLASSES:
        print(c, "choices", [selected[(f, c)] for f in folds],
              "F1 fixed/nested", round(f1(*base_counts[c]), 4), round(f1(*nested_counts[c]), 4))


if __name__ == "__main__":
    main()
