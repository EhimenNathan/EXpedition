"""Cross-modal fusion study (koz3.fuse) on training regions labelled in ≥ 2 optical/LiDAR data types.

Members (all out-of-fold): LiDAR (our heatmap U-Net, decoded at its nested-chosen setting), satellite (D-FINE +
RT-DETRv2 + heatmap fusion @0.25), Ae (heatmap, if its OOF exists). GT = every label of those regions (Li, SpOr, Ae).
Settings: twin factor alpha per direction, agreement boost beta; nested over LiDAR folds. Baseline = no fusion.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def li_features(t, dual):
    from koz3.li_train import labels, site_items, decode_site_v2
    items = {it["region"]: it for it in site_items(labels())}
    probs = pickle.load(open(os.path.join(ROOT, "work/li/oof.pkl"), "rb"))
    feats = []
    for r, p in probs.items():
        it = items[r]
        feats += decode_site_v2(p.astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=t, dual=dual)
    return feats, {r: it["fold"] for r, it in items.items()}


def df_to_feats(df, markup):
    from shapely.geometry import Point
    out = []
    for r in df.itertuples():
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(r.y / 6378137.0)) - np.pi / 2)))
        out.append({"type": "Feature", "geometry": Point(r.x, r.y).buffer(1.0 * k, 12).__geo_interface__,
                    "properties": {"class_name": r.cls, "confidence": float(r.score), "region_name": r.region, "markup_type": markup}})
    return out


def main(li_t=0.1, li_dual=0.3):
    import geopandas as gpd
    from koz3.fuse import fuse
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    from sp_assemble_oof import models
    from koz3.sp_fuse import fuse as sp_fuse
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    L = L[L["mod"].isin(["Li", "SpOr", "Ae"])]
    multi = L.groupby("region")["mod"].nunique()
    li, fold = li_features(li_t, li_dual)
    M = models(); sp = sp_fuse(list(M.values()), np.ones(len(M))); sp = sp[sp.score >= 0.25]
    members = li + df_to_feats(sp, "sp_or")
    ae_p = os.path.join(ROOT, "work/ae/oof_spdet_cands.parquet")      # deployed Ae expert: SpOr D-FINE, damaged ≥ 0.7
    if os.path.exists(ae_p):
        A = pd.read_parquet(ae_p); A = A[(A.cls == "kurgany_povrezhdennye") & (A.score >= 0.7)]
        members += df_to_feats(A, "ae")
    have = pd.Series([f["properties"]["region_name"] for f in members]).groupby(lambda i: 0).size()
    mods_of = {}
    for f in members:
        mods_of.setdefault(f["properties"]["region_name"], set()).add(f["properties"]["markup_type"])
    regions = sorted(r for r in multi.index if multi[r] >= 2 and len(mods_of.get(r, ())) >= 2 and r in fold)
    print(len(regions), "multi-modal regions with ≥ 2 predicted data types")
    folds = sorted({fold[r] for r in regions})
    gt_by_fold = {f: to_gt_geojson(L[L.region.isin([r for r in regions if fold[r] == f])]) for f in folds}
    by_fold = {f: [m for m in members if m["properties"]["region_name"] in regions and fold[m["properties"]["region_name"]] == f] for f in folds}
    pairs = [("li", "sp_or"), ("sp_or", "li"), ("li", "ae"), ("ae", "li")]
    settings = [("none",)]
    for a in (0.1, 0.2, 0.3, 0.6):
        for b in (0.0,):
            settings += [("li_out", a, b), ("li_sp", a, b), ("li_ae", a, b)]
    cache = {}

    def make(s, f):
        if s[0] == "none":
            return {"type": "FeatureCollection", "features": by_fold[f]}
        _, a, b = s
        alpha = {"both": {p: a for p in pairs}, "li_out": {p: a for p in pairs if p[0] == "li"},
                 "li_sp": {("li", "sp_or"): a}, "li_ae": {("li", "ae"): a}}[s[0]]
        return {"type": "FeatureCollection", "features": fuse(by_fold[f], alpha=alpha, beta=b)}
    C = fold_counts(make, settings, folds, gt_by_fold)
    from koz3.nested import weighted, _add_all
    base = weighted(_add_all(C[("none",)].values()))
    h, ch, o, os_, tot = nested(C)
    print(f"no fusion {base:.4f} | fusion honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {ch}")
    for st in settings:
        print("   pooled", st, round(weighted(_add_all(C[st].values())), 4))
    for c, v in tot.items():
        if v[0] + v[2]:
            print(f"   {c:24s} tp {v[0]} fp {v[1]} fn {v[2]}")


if __name__ == "__main__":
    main(*[float(x) for x in sys.argv[1:3]])
