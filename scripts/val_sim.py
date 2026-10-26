"""Validation simulator: expected leaderboard score of the deployed recipe from honest OOF predictions.

Draw validation-shaped sets of training regions (30 SpOr, 13 LAS, 4 Ae, 2 MagGr, 1 EtGr, 1 TerFl, 1 Or), pool the
experts' out-of-fold predictions and the labels of the drawn (region, data type) pairs, score with the official
pooled metric. Repeat to get a distribution. Also prints per-class F1 and each expert's leave-it-out contribution.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
COMPOSITION = {"SpOr": 30, "Li": 13, "Ae": 4, "MagGr": 2, "EtGr": 1, "TerFl": 1, "Or": 1}


def pts(df, markup, r=1.0):
    from shapely.geometry import Point
    out = []
    for row in df.itertuples():
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(row.y / 6378137.0)) - np.pi / 2)))
        out.append({"type": "Feature", "geometry": Point(row.x, row.y).buffer(r * k, 12).__geo_interface__,
                    "properties": {"class_name": row.cls, "confidence": float(row.score), "region_name": row.region, "markup_type": markup}})
    return out


def build_predictions():
    """{data type: {region: [features]}} from every expert's honest OOF output (deployed settings)."""
    from sp_assemble_oof import models
    from koz3.sp_fuse import fuse
    from koz3.li_train import labels as li_labels, site_items, decode_site_v2
    P = {}
    M = models(extra=[e for e in os.environ.get("SP_EXTRA", "").split(",") if e])
    F = fuse(list(M.values()), np.ones(len(M)))
    t_arch = float(os.environ.get("SP_ARCH_T", os.environ.get("SP_T", 0.25)))
    F = F[F.score >= np.where(F.cls == "arkhitektury", t_arch, float(os.environ.get("SP_T", 0.25)))]
    if os.environ.get("SP_FRAME"):         # framing prior: big-scene detections beyond x·half-size need score ≥ t2
        x_, t2_ = map(float, os.environ["SP_FRAME"].split(","))
        from sp_rescore_oof import scene_geometry
        G_ = scene_geometry(); keep = []
        for r in F.itertuples():
            g = G_.get(r.region, [])
            d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in g] or [(0.0, True)])
            keep.append(d[1] or d[0] <= x_ or r.score >= t2_)
        F = F[np.array(keep)]
    if os.environ.get("SP_HEDGE"):         # damaged / intact disagreement hedge
        tau = float(os.environ["SP_HEDGE"]); K_ = ("kurgany_povrezhdennye", "kurgany_tselye"); ex = []
        for r in F[F.cls.isin(K_)].itertuples():
            o = K_[1] if r.cls == K_[0] else K_[0]
            if (r.s_kt if o == K_[1] else r.s_kp) >= tau:
                ex.append((r.region, o, r.score * 0.99, r.x, r.y))
        if ex:
            F = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    if os.environ.get("SP_GATE"):          # hillfort scene gate, "swap" (scripts/sp_scene_gate_oof.py)
        g = float(os.environ["SP_GATE"]); S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet"))
        F = F[~((F.cls == "gorodishcha") & F.region.isin(set(S[S.p < g].region)))]
        a = S[S.p >= g]
        F = pd.concat([F, pd.DataFrame({"region": a.region, "cls": "gorodishcha", "score": a.p, "x": a.x, "y": a.y3857})], ignore_index=True)
    P["SpOr"] = {r: pts(g, "sp_or") for r, g in F.groupby("region")}
    items = {it["region"]: it for it in site_items(li_labels())}
    probs = pickle.load(open(os.path.join(ROOT, os.environ.get("LI_OOF", "work/li/oof.pkl")), "rb"))
    lt, ld = float(os.environ.get("LI_THR", 0.1)), float(os.environ.get("LI_DUAL", 0.3))
    P["Li"] = {r: decode_site_v2(p.astype(np.float32), items[r]["valid"], dict(items[r]["meta"], region=r), thr=lt, dual=ld) for r, p in probs.items()}
    A = pd.read_parquet(os.path.join(ROOT, "work/ae/oof_spdet_cands.parquet"))
    A = A[(A.cls == "kurgany_povrezhdennye") & (A.score >= 0.7)]
    if os.environ.get("AE_FORT_THR"):      # Ae-trained FCOS fortifications (scripts/ae_det_oof.py)
        Fo = pd.read_parquet(os.path.join(ROOT, "work/ae/oof_ae_fcos_fast_cands.parquet"))
        Fo = Fo[(Fo.cls == "fortifikatsii") & (Fo.score >= float(os.environ["AE_FORT_THR"]))]
        A = pd.concat([A, Fo[A.columns.intersection(Fo.columns)]], ignore_index=True)
    P["Ae"] = {r: pts(g, "ae") for r, g in A.groupby("region")}
    P.update(geo_predictions())
    O2 = pd.read_parquet(os.path.join(ROOT, "work/or/oof_spdet_cands.parquet")) if os.path.exists(os.path.join(ROOT, "work/or/oof_spdet_cands.parquet")) else None
    P["Or"] = {r: pts(g, "or") for r, g in O2[(O2.cls == "arkhitektury") & (O2.score >= 0.5)].groupby("region")} if O2 is not None else {}
    if os.environ.get("OR_THR"):           # Or-trained FCOS architecture, leave-one-region-out OOF
        O3 = pd.read_parquet(os.path.join(ROOT, "work/ae/oof_or_fcos_fast_cands.parquet"))
        P["Or"] = {r: pts(g, "or") for r, g in O3[(O3.cls == "arkhitektury") & (O3.score >= float(os.environ["OR_THR"]))].groupby("region")}
    return P


def geo_predictions():
    """{MagGr / EtGr / TerFl: {region: [features]}}: geophysics finds (LORO OOF) + CNN objects."""
    P = {}
    # geophysics finds: previous team's LORO OOF (p_base ≥ 0.25), unit CRS → 3857
    from pyproj import Transformer
    O = pd.read_parquet(os.path.join(ROOT, "kaggle_import/code/work/geo/finds_v3/oof.parquet"))
    O = O[O.p_base >= 0.25]
    from koz3.labels import region_utm, DATA
    finds = {}
    for reg, g in O.groupby("region"):
        tr = Transformer.from_crs(region_utm(os.path.join(DATA, reg)), 3857, always_xy=True)
        X, Y = tr.transform(g.x.values, g.y.values)
        finds[reg] = pts(pd.DataFrame({"region": reg, "cls": "finds_points", "score": g.p_base.values, "x": X, "y": Y}), "maggr", 0.25)
    # geophysics objects: CNN v1 OOF maps (objects only; the deployed 50/50 blend scores a bit higher)
    os.environ["GEO_INPUT"] = "rgb"
    from koz3.geo_train import load_all, make_preds
    gitems = {it["name"]: it for it in load_all()}
    G = pickle.load(open(os.path.join(ROOT, "work/geo/oof_probs.pkl"), "rb"))
    cache = {k: (v[0].astype(np.float32), gitems[k]) for k, v in G.items()}
    objs = {}
    for f in make_preds(cache, 9.0, 0.3, 0.5)["features"]:      # f_thr 9 → no CNN finds
        objs.setdefault(f["properties"]["region_name"], []).append(f)
    method_of = {it["meta"]["region"]: it["meta"]["method"] for _, it in cache.values()}
    for m in ("MagGr", "EtGr", "TerFl"):
        P[m] = {}
    for reg, fs in objs.items():
        P[method_of[reg]].setdefault(reg, []).extend(fs)
    for reg, fs in finds.items():
        P["MagGr"].setdefault(reg, []).extend(fs)
    return P


def main(n_draws=40, seed=0, drop=()):
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    P = build_predictions()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    mod_regions = {m: sorted(set(L[L["mod"] == m].region)) for m in COMPOSITION}
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_draws):
        preds, gts = [], []
        for m, n in COMPOSITION.items():
            pool = mod_regions[m]
            pick = rng.choice(pool, min(n, len(pool)), replace=len(pool) < n)
            for r in pick:
                if m not in drop:
                    preds += P.get(m, {}).get(r, [])
                gts.append(L[(L["mod"] == m) & (L.region == r)])
        gt = to_gt_geojson(pd.concat(gts))
        s, res = score({"type": "FeatureCollection", "features": preds}, gt)
        rows.append(dict(score=s, **{c: v["f1"] for c, v in res.items()}))
    R = pd.DataFrame(rows)
    print(f"expected validation-shaped score: mean {R.score.mean():.4f}  (5-95%: {R.score.quantile(0.05):.3f}–{R.score.quantile(0.95):.3f})")
    print(R.drop(columns="score").mean().round(3).to_string())


if __name__ == "__main__":
    import sys as _s
    if "--ablate" in _s.argv:
        import itertools
        _P = {}
        for drop in [(), ("SpOr",), ("Li",), ("MagGr", "EtGr", "TerFl"), ("Ae",), ("SpOr", "Ae"), ("Li", "Ae"), ("SpOr", "Li")]:
            print("DROP", drop or "nothing"); main(drop=drop)
    else:
        main(drop=tuple(d for d in os.environ.get("SIM_DROP", "").split(",") if d))
