"""Robust acceptance test (ROGII 2nd place, "leave-largest-contribution-out"), on per-SITE counts.

base / treat: {site: {cls: (tp, fp, fn)}}. The pooled weighted F1 gain treat − base is recomputed while greedily removing
the site whose removal shrinks the gain the most; k* = number of removals until the gain is ≤ 0. A gain that vanishes
after removing 1–2 sites hinges on luck; one that survives many removals is systematic. Also reports the share of sites
that improve / worsen and a site bootstrap of the gain.

  python scripts/robust_accept.py li_prior        # deployed LiDAR blend: no prior vs neighbourhood prior (v14)
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from koz3.nested import weighted, _add_all  # noqa: E402


def curve(base, treat):
    sites = sorted(set(base) & set(treat)); left = list(sites); out = []
    gain = lambda S: weighted(_add_all(treat[s] for s in S)) - weighted(_add_all(base[s] for s in S))
    out.append((None, gain(left)))
    while len(left) > 1:
        s = min(left, key=lambda s: gain([t for t in left if t != s]))
        left.remove(s); out.append((s, gain(left)))
    return out


def report(name, base, treat, n_boot=2000, seed=0):
    c = curve(base, treat); g0 = c[0][1]
    k = next((i for i, (_, g) in enumerate(c) if g <= 0), len(c))
    sites = sorted(set(base) & set(treat))
    per = {s: weighted(treat[s]) - weighted(base[s]) for s in sites}
    up, dn = sum(v > 1e-9 for v in per.values()), sum(v < -1e-9 for v in per.values())
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        S = list(rng.choice(sites, len(sites)))
        bs.append(weighted(_add_all(treat[s] for s in S)) - weighted(_add_all(base[s] for s in S)))
    bs = np.array(bs)
    print(f"[{name}] {len(sites)} sites  gain {g0:+.4f}  k*={k}  sites up/down/same {up}/{dn}/{len(sites) - up - dn}  "
          f"bootstrap P(gain>0)={np.mean(bs > 0):.2f}  90% CI [{np.quantile(bs, .05):+.4f}, {np.quantile(bs, .95):+.4f}]")
    print("   removal path:", ", ".join(f"{s}:{g:+.4f}" for s, g in c[1:min(len(c), 8)]))
    return g0, k


def site_counts(make, settings, sites, gt, classes=None):
    from koz3.score import score
    C = {s: {} for s in settings}
    for st in settings:
        for r in sites:
            _, res = score(make(st, r), gt[r], classes)
            C[st][r] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
    return C


def li_prior():
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    import li_compare_oof as lc
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = {nm: lc._pk(f"work/li/oof_{nm}.pkl.zst") for nm in ("ens", "aug", "augx")}
    B = {r: ((P["ens"][r].astype(np.float32) + P["aug"][r] + P["augx"][r]) / 3).astype(np.float32) for r in P["ens"]}
    sites = sorted(B); gt = {r: to_gt_geojson(L[L.region == r]) for r in sites}
    pri = {"R": 200.0, "K": 3, "snb": 0.2}

    def make(st, r):
        it = items[r]
        return {"type": "FeatureCollection", "features": decode_site_v2(B[r], it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05,
                                                                       comp_thr=0.5, prior=pri if st == "prior" else None)}
    C = site_counts(make, ("base", "prior"), sites, gt, CLS)
    report("LiDAR nbr prior (v14, platform +)", C["base"], C["prior"])


def sat(a="dfine,rtdetr,hm,dfnodn,balq,balq3", b="dfine,rtdetr,hm,dfnodn,balq,balq3,mos"):
    """Deployed satellite pipeline (fusion, 0.25 / arch 0.15, framing prior, hedge 0.4, hillfort gate) per region:
    member set a (base) vs b (treat)."""
    import geopandas as gpd
    import pandas as pd
    from koz3.labels import to_gt_geojson
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_fusion4_oof import gate
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    A, Bn = a.split(","), b.split(",")
    M = models(extra=sorted({n for n in A + Bn if n not in ("dfine", "rtdetr", "hm")}))
    regions = sorted(set.intersection(*[set(m.region) for m in M.values()])); RF = region_folds()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()

    def pipeline(names):
        F = fuse_cls([M[n][M[n].region.isin(regions)] for n in names])
        F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True); keep = []
        for r in F.itertuples():
            d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
            keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
        F = F[np.array(keep)]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
        return pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    P = {"a": pipeline(A), "b": pipeline(Bn)}
    C = site_counts(lambda st, r: fc(gate(P[st][P[st].region == r], S[S.region == r])), ("a", "b"), regions, gt)
    report(f"satellite {a} → {b}", C["a"], C["b"])


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
