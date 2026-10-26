"""Per-sheet anatomy of the fused Ae kurgan detector (work/ae_cluster_counts.pkl from scripts/ae_cluster_oof.py):
TP / FP per training Ae region at fused score ≥ t, against the region's label density — do detectors work on dense
cemetery sheets (the public Ae regions average ~140 objects) and fail elsewhere?
  python scripts/ae_sheet_anatomy.py
"""
import os
import pickle
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(t=0.3):
    import geopandas as gpd
    C = pickle.load(open(os.path.join(ROOT, "work/ae_cluster_counts.pkl"), "rb"))
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); A = L[L["mod"] == "Ae"]
    n = A.groupby("region").size(); nk = A[A.class_name.isin(K)].groupby("region").size()
    rows = []
    for k, v in C.items():
        if k[0] == "Ae" and k[2] == (t, None):
            tp = sum(v.get(c, (0, 0, 0))[0] for c in K); fp = sum(v.get(c, (0, 0, 0))[1] for c in K)
            rows.append((k[1], int(n.get(k[1], 0)), int(nk.get(k[1], 0)), tp, fp))
    D = pd.DataFrame(rows, columns=["region", "gt", "gt_k", "tp", "fp"])
    D["share"] = (D.tp / (D.tp + D.fp).clip(lower=1)).round(3)
    print(D.sort_values("tp", ascending=False).to_string(index=False))
    dense = D[D.gt_k >= 80]
    print(f"\ndense sheets (≥ 80 kurgan labels): TP {dense.tp.sum()} FP {dense.fp.sum()} share {dense.tp.sum() / max(dense.tp.sum() + dense.fp.sum(), 1):.3f}")
    print(f"other sheets: TP {D.tp.sum() - dense.tp.sum()} FP {D.fp.sum() - dense.fp.sum()}")


if __name__ == "__main__":
    main(*[float(a) for a in sys.argv[1:]])
