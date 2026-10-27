"""Satellite illumination-consistency rescoring (koz3.shading), nested + robust, on the deployed v20 satellite pipeline.

Rule: for kurgan candidates, fused score' = score × (1 + λ · sh_cos · sh_R) before the deployed thresholds, framing
prior, damaged/intact hedge and hillfort gate + snap. sh_cos = agreement of the candidate's brightness dipole with the
leave-one-out scene sun azimuth (other confident candidates of the same image), sh_R = concentration of that azimuth
(≈ 0 when the image has no common shading, which turns the rule off by itself). λ = 0 reproduces deployment.
λ is chosen on the other folds (koz3.nested); the per-site leave-largest-contribution-out test reports k*.

  python scripts/sp_shading_oof.py            # features cached in work/spor/oof_shading.parquet
Accept only if: nested honest > deployed, no outer fold regresses, k* ≥ 3, bootstrap P(gain>0) ≥ 0.8.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def shading_features(F, cache):
    """One row per fused candidate: sh_cos, sh_R, sh_mag, sh_radius (NaN-free; 0 when no scene covers it)."""
    if os.path.exists(cache):
        S = pd.read_parquet(cache)
        if len(S) == len(F) and np.allclose(S.x.values, F.x.values):
            return S
    import rasterio
    from pyproj import Transformer
    from koz3.labels import DATA
    from koz3.shading import scene_features
    crs_tab = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site")
    out = pd.DataFrame({"x": F.x.values, "sh_cos": 0.0, "sh_R": 0.0, "sh_mag": 0.0, "sh_radius": 0.0})
    for region, idx in F.groupby("region").groups.items():
        idx = np.asarray(list(idx)); todo = np.ones(len(idx), bool)
        for path in sorted(glob.glob(os.path.join(DATA, region, "04_*", "*.tif"))):
            with rasterio.open(path) as ds:
                crs = f"EPSG:{int(crs_tab.loc[region, 'json_epsg'])}" if region in crs_tab.index else ds.crs.to_string()
                tr = Transformer.from_crs(3857, crs, always_xy=True)
                X, Y = tr.transform(F.x.values[idx], F.y.values[idx])
                rows, cols = rasterio.transform.rowcol(ds.transform, X, Y)
                rows, cols = np.asarray(rows), np.asarray(cols)
                inside = todo & (rows >= 0) & (cols >= 0) & (rows < ds.height) & (cols < ds.width)
                if not inside.any():
                    continue
                a = ds.read(indexes=list(range(1, min(3, ds.count) + 1))).astype(np.float32)
                gray = a.mean(0)
                res = abs(ds.transform.a) * (111320.0 if not ds.crs.is_projected else 1.0)
                sel = np.flatnonzero(inside)
                f = scene_features(gray, rows[sel], cols[sel], res, F.score.values[idx[sel]])
                for k, v in f.items():
                    out.loc[idx[sel], k] = v
                todo[sel] = False
        print(f"shading {region}: {int((~todo).sum())}/{len(idx)} candidates in a scene", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_parquet(cache)
    return out


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    from robust_accept import site_counts, report
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds()
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    SH = shading_features(F0, os.path.join(ROOT, "work/spor/oof_shading.parquet"))
    kur = F0.cls.isin(K).values
    print(f"{len(regions)} regions, {len(F0)} fused candidates, {int((SH.sh_R > 0).sum())} with shading features")
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0

    def pipeline(lam, regs):
        """Deployed v20 satellite decoding (fusion 0.25 / arch 0.15, framing prior 0.5 / 0.35, hedge 0.4, gate 0.1,
        hillfort snap ρ 0.2 s 0.02) on shading-rescaled kurgan scores."""
        F = F0.copy()
        F.loc[kur, "score"] = F.score[kur] * (1 + lam * SH.sh_cos[kur] * SH.sh_R[kur])
        F = F[F.region.isin(regs)]
        F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)]; keep = []
        for r in F.itertuples():
            d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
            keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
        F = F[np.array(keep, bool)]
        ex = [(r.region, K[1] if r.cls == K[0] else K[0], r.score * 0.99, r.x, r.y) for r in F[F.cls.isin(K)].itertuples()
              if (r.s_kt if r.cls == K[0] else r.s_kp) >= 0.4]
        G = pd.concat([F[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        Sx = Sg[Sg.region.isin(regs)]
        G = G[~((G.cls == "gorodishcha") & G.region.isin(set(Sx[Sx.p < 0.1].region)))]
        rows = []
        for r in Sx[Sx.p >= 0.1].itertuples():
            cx, cy = r.x, r.y3857
            c = GC[(GC.region == r.region) & (GC.score >= 0.02)]
            c = c[np.hypot(c.x - cx, c.y - cy) <= 0.2 * half[r.Index]]
            if len(c):
                b = c.loc[c.score.idxmax()]; cx, cy = b.x, b.y
            rows.append((r.region, "gorodishcha", r.p, cx, cy))
        return pd.concat([G, pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)

    folds = sorted({RF[r] for r in regions})
    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    C = fold_counts(lambda lam, f: fc(pipeline(lam, [r for r in regions if RF[r] == f])), LAMBDAS, folds, gt)
    for lam in LAMBDAS:
        tot = _add_all(C[lam].values())
        per_fold = " ".join(f"{weighted(C[lam][f]):.4f}" for f in folds)
        print(f"λ={lam:<4} pooled {weighted(tot):.4f}  folds [{per_fold}]  "
              + " ".join(f"{c[:8]} {tuple(int(v) for v in tot[c])}" for c in K if c in tot), flush=True)
    h, ch, o, os_, _ = nested(C)
    base = weighted(_add_all(C[0.0].values()))
    print(f"deployed (λ=0) {base:.4f} | nested honest {h:.4f} ({h - base:+.4f}) oracle {o:.4f} @ λ={os_} chosen {ch}")
    worst = min(weighted(C[ch[f]][f]) - weighted(C[0.0][f]) for f in folds)
    print(f"worst outer-fold change {worst:+.4f}")
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    lam = max(set(ch.values()), key=list(ch.values()).count)
    CS = site_counts(lambda s, r: fc(pipeline(s, [r])), (0.0, lam), regions, gtr)
    report(f"shading λ={lam}", CS[0.0], CS[lam])


if __name__ == "__main__":
    main()
