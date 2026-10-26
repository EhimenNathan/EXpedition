"""Coarse satellite branch (D-FINE at 2 m/px on large objects ≥ 40 m) added BESIDE the v12 satellite pipeline (6-member
fusion, arch 0.15, framing prior, damaged/intact hedge 0.4, hillfort gate): coarse detections with score ≥ t_c (framing
prior: far ones need ≥ t_c + 0.1) are added where the pipeline has no same-family detection within d metres. Nested."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc, to3857
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry
from sp_dual_oof import fuse_cls
K = ("kurgany_povrezhdennye", "kurgany_tselye"); fam = lambda c: "k" if c in K else c


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
    M = models(extra=["dfnodn", "balq", "balq3"]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    geom = scene_geometry()
    def frame(D, t2):
        keep = []
        for r in D.itertuples():
            d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
            keep.append(d[1] or d[0] <= 0.5 or r.score >= t2)
        return D[np.array(keep, bool)] if len(D) else D
    F = fuse_cls([M[n] for n in names]); F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True)
    F = frame(F, 0.35).reset_index(drop=True)
    ex = []
    for r in F[F.cls.isin(K)].itertuples():
        o = K[1] if r.cls == K[0] else K[0]
        if (r.s_kt if o == K[1] else r.s_kp) >= 0.4:
            ex.append((r.region, o, r.score * 0.99, r.x, r.y))
    B = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    crs = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site"); crs_of = {s: f"EPSG:{int(e)}" for s, e in crs["json_epsg"].items()}
    Cc = to3857(pd.read_parquet(os.path.join(ROOT, "kaggle_import/code/work/sp_runs/sp_dfine_coarse/oof_pred0123.parquet")), crs_of)
    Cc = Cc[Cc.region.isin(regions)]
    print("coarse candidates", len(Cc), "max score", round(Cc.score.max(), 3))

    def make(s, f):
        tc, dm = s
        G = B[B.region.map(RF) == f]
        if tc < 9:
            c = frame(Cc[(Cc.region.map(RF) == f) & (Cc.score >= tc)], tc + 0.1)
            add = []
            for reg, g in c.groupby("region"):
                base = G[G.region == reg]
                for r in g.itertuples():
                    same = base[base.cls.map(fam) == fam(r.cls)]
                    if not len(same) or np.min(np.hypot(same.x - r.x, same.y - r.y)) > dm:
                        add.append((r.region, r.cls, r.score, r.x, r.y))
            if add:
                G = pd.concat([G, pd.DataFrame(add, columns=G.columns)], ignore_index=True)
        return fc(gate(G, S[S.region.map(RF) == f]))
    SS = [(9, 0)] + [(tc, dm) for tc in (0.3, 0.4, 0.5, 0.6) for dm in (15, 30, 60)]
    C = fold_counts(make, SS, folds, gt)
    for s in SS:
        tot = _add_all(C[s].values())
        print(f"t_c {s[0]} d {s[1]}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot['kurgany_povrezhdennye'])}  gor {tuple(int(v) for v in tot['gorodishcha'])}")
    h, ch, o, os_, _ = nested(C)
    print(f"v12 satellite {weighted(_add_all(C[(9, 0)].values())):.4f} | + coarse nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")


if __name__ == "__main__":
    main()
