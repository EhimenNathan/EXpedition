"""Exact-matcher localization ceiling of raw satellite OOF candidates.

Counts whether each GT object has *any* held-out raw candidate that could
match it. This is an oracle coverage diagnostic, not a deployable score.
"""
import json
import os
import sys
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from koz3.score import CALC
from koz3.labels import to_gt_geojson
from sp_assemble_oof import models
from sp_fusion_oof import fc

NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
TS = (0.02, 0.05, 0.1, 0.25)

def main():
    M = models(extra=list(NAMES[3:]))
    regs = sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    L = L[(L["mod"] == "SpOr") & L.region.isin(regs)]
    A = pd.concat([M[n][M[n].region.isin(regs)].assign(model=n) for n in NAMES], ignore_index=True)
    A = A[A.score >= min(TS)]
    out = defaultdict(lambda: defaultdict(lambda: dict(gt=0, hits={str(t): 0 for t in TS})))
    for i, region in enumerate(regs):
        gt = CALC._collect_by_class(to_gt_geojson(L[L.region == region]), preserve_points=True)
        pred = CALC._collect_by_class(fc(A[A.region == region]), preserve_points=False)
        for cls, gts in gt.items():
            ps = pred.get(cls, [])
            polys = [p for p, _ in ps]
            scores = [s for _, s in ps]
            tree = STRtree(polys) if polys else None
            v = out[region][cls]
            v["gt"] += len(gts)
            for g, _ in gts:
                hits = {t: False for t in TS}
                if tree is not None:
                    query = g.buffer(CALC.r_tolerance + 0.1) if g.geom_type == "Point" else g
                    for j in tree.query(query):
                        p, s = polys[j], scores[j]
                        if CALC._match_quality(p, g)[0]:
                            for t in TS:
                                hits[t] |= s >= t
                            if all(hits.values()):
                                break
                for t in TS:
                    v["hits"][str(t)] += int(hits[t])
        if (i + 1) % 25 == 0:
            print("regions", i + 1, "/", len(regs), flush=True)
    total = defaultdict(lambda: dict(gt=0, hits={str(t): 0 for t in TS}))
    for r in out.values():
        for cls, v in r.items():
            total[cls]["gt"] += v["gt"]
            for t in TS:
                total[cls]["hits"][str(t)] += v["hits"][str(t)]
    obj = {"per_region": out, "total": total}
    dest = os.path.join(os.path.dirname(__file__), "sp_raw_candidate_oracle.json")
    with open(dest, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(json.dumps(total, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
