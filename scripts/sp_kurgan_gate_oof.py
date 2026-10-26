"""Kurgan scene gate on satellite small scenes: a frozen ConvNeXt-T (IN-22k) linear probe predicts "scene contains
kurgans" (OOF on the shared site folds, AUC 0.798); kurgan detections in small scenes with p < g are dropped (or scaled).
On top of the v12 satellite pipeline (5-member fusion, global 0.25 / architecture 0.15, hillfort gate). Nested over g."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc, fuse
from sp_fusion4_oof import gate
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(col=os.environ.get("PK", "pk_cnx")):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    M = models(extra=["dfnodn", "balq"]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_kurgan.parquet")); S = S[S.region.isin(regions)]
    F = fuse([M[n] for n in ("dfine", "rtdetr", "hm", "dfnodn", "balq")], np.ones(5))
    F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)]
    pk = S.groupby("region")[col].max()                     # region kurgan probability (max over its small scenes)
    small = set(S.region)

    def make(s, f):
        g = s[0]
        G = F[F.region.map(RF) == f]
        if g > 0:
            low = {r for r in small if pk[r] < g}
            G = G[~(G.cls.isin(K) & G.region.isin(low))]
        return fc(gate(G, S[S.region.map(RF) == f]))
    SS = [(0.0,)] + [(g,) for g in (0.1, 0.2, 0.3, 0.4, 0.5)]
    C = fold_counts(make, SS, folds, gt)
    for s in SS:
        print("pooled g", s[0], round(weighted(_add_all(C[s].values())), 4))
    h, ch, o, os_, tot = nested(C)
    print(f"no gate {weighted(_add_all(C[(0.0,)].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {ch}")


if __name__ == "__main__":
    main()
