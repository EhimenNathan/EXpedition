"""Satellite fusion with the locally trained D-FINE-L without denoising queries (seed 1, same folds) as a 4th member.
Nested over member set × fused threshold; every variant carries the deployed hillfort gate ("swap", g = 0.1).

  python scripts/sp_fusion4_oof.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds  # noqa: E402
from sp_fusion_oof import fc, fuse  # noqa: E402


def gate(F, S, g=0.1):
    F = F[~((F.cls == "gorodishcha") & F.region.isin(set(S[S.p < g].region)))]
    a = S[S.p >= g]
    return pd.concat([F, pd.DataFrame({"region": a.region, "cls": "gorodishcha", "score": a.p, "x": a.x, "y": a.y3857})], ignore_index=True)


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    M = models(extra=["dfnodn", "balq"])
    regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    sets = [("dfine", "rtdetr", "hm", "dfnodn"), ("balq",), ("dfnodn",), ("dfine", "rtdetr", "hm", "balq"),
            ("dfine", "rtdetr", "hm", "dfnodn", "balq"), ("dfine", "rtdetr", "dfnodn", "balq")]
    fused = {s: fuse([M[n] for n in s], np.ones(len(s))) for s in sets}
    TH = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)

    def make(st, f):
        s, t = st
        F = fused[s]; F = F[(F.score >= t) & (F.region.map(RF) == f)]
        return fc(gate(F, S[S.region.map(RF) == f]))
    C = fold_counts(make, list(itertools.product(sets, TH)), folds, gt)
    for s in sets:
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[0] == s})
        print(f"{'+'.join(s):30s} honest {h:.4f}  oracle {o:.4f} @ t={os_[1]}  chosen t {[c[1] for c in ch.values()]}  "
              + " ".join(f"{c[:6]} {2 * v[0] / max(2 * v[0] + v[2] + 0.5 * v[1], 1):.3f}" for c, v in tot.items() if v[0] + v[2]))
    h, ch, o, os_, tot = nested(C)
    print(f"ALL (set and t nested)          honest {h:.4f}  chosen {ch}")


if __name__ == "__main__":
    main()
