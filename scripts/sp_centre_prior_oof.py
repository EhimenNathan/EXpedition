"""Satellite scene-framing prior (nested): scenes are framed on the site and labels exist around it, so in BIG scenes a
kurgan farther than x (fraction of the scene half-size) from the nearest scene centre must reach score t2 (t2 = 1 drops
it). On the v12 pipeline (5-member fusion, global 0.25 / architecture 0.15, hillfort gate). Scene geometry comes from
the image itself (deployable)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds  # noqa: E402
from sp_fusion_oof import fc, fuse  # noqa: E402
from sp_fusion4_oof import gate  # noqa: E402
from sp_rescore_oof import scene_geometry  # noqa: E402

K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq")
    M = models(extra=["dfnodn", "balq"]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    F = fuse([M[n] for n in names], np.ones(len(names)))
    F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True)
    geom = scene_geometry()
    rel, small = [], []
    for r in F.itertuples():
        g = geom.get(r.region, [])
        d = [(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in g] or [(0.0, True)]
        b = min(d); rel.append(b[0]); small.append(b[1])
    F["d_rel"], F["small"] = rel, small

    def make(s, f):
        x, t2, cls_set = s
        G = F[F.region.map(RF) == f]
        if x < 9:
            far = (~G.small) & (G.d_rel > x) & (G.cls.isin(cls_set))
            G = G[~far | (G.score >= t2)]
        return fc(gate(G[["region", "cls", "score", "x", "y"]], S[S.region.map(RF) == f]))
    ALL = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii")
    SS = [(9, 0, K)] + [(x, t2, ALL) for x in (0.25, 0.35, 0.5, 0.6) for t2 in (0.3, 0.35, 0.4)]
    C = fold_counts(make, SS, folds, gt)
    for s in SS:
        tot = _add_all(C[s].values())
        print(f"x {s[0]} t2 {s[1]} {'kurgans' if s[2] == K else 'all'}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot['kurgany_povrezhdennye'])}")
    h, ch, o, os_, _ = nested(C)
    print(f"base {weighted(_add_all(C[SS[0]].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_[:2]}) chosen {[c[:2] for c in ch.values()]}")


if __name__ == "__main__":
    main()
