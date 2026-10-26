"""Satellite hillfort gate: the whole-scene hillfort classifier (koz3/scene_cls.py, OOF on the same site folds) adds a
gorodishcha point at the centre of small scenes it is confident about, on top of the deployed fusion (equal weights,
threshold 0.25). Nested over gate threshold × mode × confidence; compared with the fusion alone and the ungated
centre prior.

  mode "add":   always add the centre point when p ≥ g
  mode "fill":  add only when the fusion has no gorodishcha in that region
  mode "swap":  as "add", and drop the fusion's own gorodishcha in regions where p < g
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds  # noqa: E402
from sp_fusion_oof import fc, fuse  # noqa: E402


def main(t=0.25):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    M = models()
    regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    Fz = fuse([M[n][M[n].region.isin(regions)] for n in M], np.ones(len(M)))
    Fall = Fz[Fz.cls == "gorodishcha"]
    Fz = Fz[Fz.score >= t]
    S = pd.read_parquet(os.path.join(ROOT, os.environ.get("SCENE_OOF", "work/spor/oof_scene_cls.parquet")))
    S = S[S.region.isin(regions)]
    print(len(regions), "regions;", len(S), "small scenes in them,", int(S.y.sum()), "with a hillfort")
    has_gor = set(Fz[Fz.cls == "gorodishcha"].region)

    def make(s, f):
        mode, g, conf = s
        F = Fz[Fz.region.map(RF) == f]
        if mode == "none":
            return fc(F)
        c = S[S.region.map(RF) == f]
        if mode == "prior":                       # ungated centre on every small scene
            add = c.assign(p=0.05)
        else:
            add = c[c.p >= g]
            if mode == "fill":
                add = add[~add.region.isin(has_gor)]
            if mode == "swap":
                low = set(c[c.p < g].region)
                F = F[~((F.cls == "gorodishcha") & F.region.isin(low))]
        rows = pd.DataFrame({"region": add.region, "cls": "gorodishcha",
                             "score": add.p if conf == "p" else conf, "x": add.x, "y": add.y3857})
        if mode.startswith("best"):               # gated scene → the fusion's best hillfort candidate (any score)
            F = F[~((F.cls == "gorodishcha") & F.region.isin(set(c[c.p < g].region)))] if mode == "best_swap" else F
            b = Fall[Fall.region.isin(set(add.region))].sort_values("score").groupby("region").tail(1)
            pp = add.groupby("region").p.max()
            rows = pd.concat([pd.DataFrame({"region": b.region, "cls": "gorodishcha", "score": b.region.map(pp), "x": b.x, "y": b.y}),
                              rows[~rows.region.isin(set(b.region))]])
        return fc(pd.concat([F, rows]) if len(rows) else F)
    settings = [("none", None, None), ("prior", None, None)]
    settings += [(m, g, cf) for m in ("add", "fill", "swap", "best", "best_swap") for g in (0.02, 0.05, 0.1, 0.2, 0.3) for cf in ("p", 0.05)]
    C = fold_counts(make, settings, folds, gt)
    for name, keys in (("fusion only", [settings[0]]), ("+ centre prior", [settings[1]]),
                       ("+ gate add", [s for s in settings if s[0] == "add"]),
                       ("+ gate fill", [s for s in settings if s[0] == "fill"]),
                       ("+ gate swap", [s for s in settings if s[0] == "swap"]),
                       ("+ gate best", [s for s in settings if s[0] == "best"]),
                       ("+ gate best_swap", [s for s in settings if s[0] == "best_swap"]),
                       ("ALL", settings)):
        h, ch, o, os_, tot = nested({k: C[k] for k in keys})
        gor = tot.get("gorodishcha", (0, 0, 0))
        print({c: tuple(int(x) for x in v) for c, v in tot.items()}) if name == "ALL" else None
        print(f"{name:16s} honest {h:.4f}  oracle {o:.4f} @ {os_}  gor tp/fp/fn {tuple(int(v) for v in gor)}  chosen {ch}")


if __name__ == "__main__":
    main()
