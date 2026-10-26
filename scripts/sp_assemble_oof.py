"""Satellite OOF assembly: fused detectors + hillfort localiser + centre fallback, scored together.

Hillfort rule (per chip): localiser components with score ≥ g_thr are emitted as gorodishcha (confidence = score);
if a small scene has none, but its localiser max ≥ c_thr, the scene centre is emitted with confidence c_conf.
Detector gorodishcha candidates are kept or dropped (keep_det_gor) — the localiser may simply be better.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from koz3.score import score  # noqa: E402
from sp_fusion_oof import fc, fuse, load_theirs, to3857  # noqa: E402


def models(extra=()):
    crs = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site")
    crs_of = {s: f"EPSG:{int(e)}" for s, e in crs["json_epsg"].items()}
    D = to3857(load_theirs(os.path.join(ROOT, "kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l")), crs_of)
    R = to3857(load_theirs(os.path.join(ROOT, "kaggle_import/out/koz3-sp-rtdetr/sp_runs/sp_rtdetrv2_r50")), crs_of)
    H = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_hm_cands.parquet"))
    out = {"dfine": D, "rtdetr": R, "hm": H}
    for name in extra:
        p = os.path.join(ROOT, f"work/spor/oof_det_{name}_cands.parquet")
        if os.path.exists(p):
            out[name] = pd.read_parquet(p)[["region", "cls", "score", "x", "y"]]
    return out


def hillfort_rows(G, g_thr, c_thr, c_conf, comp_thr=0.5):
    comp = G[(G.thr == comp_thr) & (G.score >= g_thr)]
    rows = [(r.region, "gorodishcha", r.score, r.x, r.y) for r in comp.itertuples()]
    has = set(comp.region)
    cen = G[(G.thr == -1) & (G.kind == "small") & (G.scene_max >= c_thr) & ~G.region.isin(has)]
    rows += [(r.region, "gorodishcha", c_conf, r.x, r.y) for r in cen.itertuples()]
    return pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])


def region_folds():
    import json
    f = os.path.join(ROOT, "kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/folds.json")
    return {k.rsplit("__", 1)[0]: int(v) for k, v in json.load(open(f, encoding="utf-8")).items()}


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    M = models(extra=sys.argv[1:])
    regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt_by_fold = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    G = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_gor_cands.parquet")) if os.path.exists(os.path.join(ROOT, "work/spor/oof_gor_cands.parquet")) else None
    names = list(M)
    print(len(regions), "regions; models", names, "folds", folds)
    weight_sets = {"equal": np.ones(len(names))}
    if "dfine" in names:
        weight_sets["dfine_only"] = np.array([1.0 if n == "dfine" else 0.0 for n in names])
    fused = {k: fuse([M[n] for n in names], w) for k, w in weight_sets.items()}
    settings = []
    for wk in weight_sets:
        for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
            settings.append((wk, t, True, None, None))
            if G is not None:
                settings.append((wk, t, True, "centre", None))      # centre prior on every small scene
                settings.append((wk, t, False, "centre", None))

    def make(sett, f):
        wk, t, keep, g_thr, c_thr = sett
        Fz = fused[wk]; Fz = Fz[(Fz.score >= t) & (Fz.region.map(RF) == f)]
        parts = [Fz if keep else Fz[Fz.cls != "gorodishcha"]]
        if g_thr == "centre":
            c = G[(G.thr == -1) & (G.kind == "small") & (G.region.map(RF) == f)]
            parts.append(pd.DataFrame({"region": c.region, "cls": "gorodishcha", "score": 0.05, "x": c.x, "y": c.y}))
        elif g_thr is not None:
            parts.append(hillfort_rows(G[G.region.map(RF) == f], g_thr, c_thr, 0.05))
        return fc(pd.concat([q for q in parts if len(q)]))
    C = fold_counts(make, settings, folds, gt_by_fold)
    for wk in weight_sets:
        sub = {k: v for k, v in C.items() if k[0] == wk and k[3] is None}
        h, ch, o, os_, _ = nested(sub)
        print(f"{wk:10s} detectors only   honest {h:.4f}  oracle {o:.4f} @ {os_[1]}  chosen {ch}")
    h, ch, o, os_, tot = nested(C)
    print(f"ALL settings      honest {h:.4f}  oracle {o:.4f} @ {os_}")
    print("chosen per fold:", ch)
    print("honest per-class:", {c: v for c, v in tot.items()})


if __name__ == "__main__":
    main()
