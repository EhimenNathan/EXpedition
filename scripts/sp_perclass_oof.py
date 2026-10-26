"""Per-class fused thresholds (nested) for the 5-member satellite fusion + hillfort gate: the metric scores classes
independently, so each class's threshold is chosen on the other folds."""
import os, sys, itertools
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc, fuse
from sp_fusion4_oof import gate
from ae_det_oof import f1


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts
    from koz3.score import WEIGHTS
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq") + (("balq3",) if os.environ.get("SP_MEMBERS") == "6" else ())
    M = models(extra=[n for n in names if n not in ("dfine", "rtdetr", "hm")]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    F = fuse([M[n] for n in names], np.ones(len(names)))
    TH = (0.15, 0.2, 0.25, 0.3, 0.35)
    C = fold_counts(lambda t, f: fc(gate(F[(F.score >= t[0]) & (F.region.map(RF) == f)], S[S.region.map(RF) == f])), [(t,) for t in TH], folds, gt)
    CLS = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
    for mode in ("global", "per-class"):
        tot = {}; chosen = {}
        for f in folds:
            oth = lambda c, t: np.sum([C[(t,)][g].get(c, (0, 0, 0)) for g in folds if g != f], 0)
            if mode == "global":
                tg = max(TH, key=lambda t: sum(WEIGHTS[c] * f1(*oth(c, t)) for c in CLS))
            for c in CLS:
                t = tg if mode == "global" else max(TH, key=lambda t: f1(*oth(c, t)))
                chosen.setdefault(c, []).append(t); tot[c] = np.add(tot.get(c, (0, 0, 0)), C[(t,)][f].get(c, (0, 0, 0)))
        pres = [c for c in CLS if tot[c][0] + tot[c][2]]
        print(f"{mode:9s} honest {sum(WEIGHTS[c] * f1(*tot[c]) for c in pres) / sum(WEIGHTS[c] for c in pres):.4f}  " + "  ".join(f"{c[:8]} {f1(*tot[c]):.3f} t{chosen[c]}" for c in pres))


if __name__ == "__main__":
    main()
