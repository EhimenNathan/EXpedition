"""Anatomy of satellite kurgan FPs on the deployed v14 pipeline (OOF): for each emitted kurgan point not matched,
distance to the nearest kurgan GT polygon (any class / same class) and whether that GT was itself missed."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
from sp_assemble_oof import models, region_folds
from sp_rescore_oof import scene_geometry
from sp_dual_oof import fuse_cls, K
names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
M = models(extra=list(names[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions) & L.class_name.isin(K)]
F = fuse_cls([M[n][M[n].region.isin(regions)] for n in names]); F = F[F.cls.isin(K) & (F.score >= 0.25)].reset_index(drop=True)
geom = scene_geometry(); keep = []
for r in F.itertuples():
    d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
    keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
F = F[np.array(keep)].reset_index(drop=True)
rows = []
for reg, g in F.groupby("region"):
    gt = L[L.region == reg]; polys = list(gt.geometry); cls = list(gt.class_name)
    if not polys:
        rows += [(reg, r.cls, r.score, np.inf, np.inf, False, False) for r in g.itertuples()]; continue
    k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(g.y.mean() / 6378137.0)) - np.pi / 2)))
    t = STRtree(polys)
    for r in g.itertuples():
        p = Point(r.x, r.y); inside = [j for j in t.query(p) if polys[j].contains(p)]
        j = t.nearest(p); d = polys[j].distance(p) / k
        rows.append((reg, r.cls, r.score, d, polys[j].area / k / k, bool(inside), bool(inside) and any(cls[i] == r.cls for i in inside)))
D = pd.DataFrame(rows, columns=["region", "cls", "score", "dist_m", "gt_area", "inside_any", "inside_same"])
print(len(D), "emitted kurgan points (deployed thresholds)")
D["cat"] = np.select([D.inside_same, D.inside_any, D.dist_m <= 3, D.dist_m <= 8, D.dist_m <= 30], ["hit-same", "inside-other-class", "≤3 m outside", "3–8 m", "8–30 m"], "far >30 m")
print(D.cat.value_counts().to_string())
print("score median by category:", D.groupby("cat").score.median().round(3).to_dict())
D.to_parquet("work/sp_fp_anatomy.parquet")
