"""Satellite DINOv3-SAT verifier on the deployed v21 satellite OOF candidates, nested + robust (decisive test for
experiments/claude/sp_dinov3_probe.py).

Probe evidence (19 screenshots, leave-scene-out, frozen ViT-L/16 SAT-493M + linear probe on LoG blob candidates):
AUC 0.749 pooled / 0.856 per scene vs 0.576 / 0.582 for blob appearance. The question here is whether that signal
adds value ON TOP of the deployed detector scores.

Steps:
1. fused deployed candidates (koz3.sp_fuse via sp_dual_oof.fuse_cls, 6 members) with fused score ≥ 0.10;
2. per candidate, a 64 m × 64 m chip at 0.5 m from its satellite scene → frozen DINOv3 CLS + mean patch token
   (cached in work/spor/oof_dinov3.npz; GPU if available);
3. per satellite fold, a logistic verifier trained on the other folds (label: candidate inside a GT polygon of the
   same class family) → out-of-fold verifier probability v;
4. kurgan scores rescaled score' = (1 − λ)·score + λ·v before the deployed decoding (thresholds, framing prior, hedge,
   hillfort gate + snap); λ = 0 reproduces deployment;
5. λ chosen on the other folds (koz3.nested); per-fold change; per-region robust acceptance (k*, bootstrap).

Requires an accepted DINOv3 licence and HF_TOKEN.
  python scripts/sp_dinov3_verifier_oof.py
Accept only if nested honest > deployed, no outer fold regresses, k* ≥ 3, bootstrap P(gain>0) ≥ 0.8.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "experiments", "claude"))
NAMES = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
LAMBDAS = (0.0, 0.15, 0.3, 0.5, 0.7)
MODEL = os.environ.get("DINOV3_MODEL", "facebook/dinov3-vitl16-pretrain-sat493m")


def embeddings(F, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z["E"]) == len(F):
            return z["E"], z["ok"]
    import rasterio
    import torch
    from pyproj import Transformer
    from transformers import AutoModel, AutoImageProcessor
    from koz3.labels import DATA
    from sp_dinov3_probe import chip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = os.environ.get("HF_TOKEN")
    proc = AutoImageProcessor.from_pretrained(MODEL, token=tok)
    model = AutoModel.from_pretrained(MODEL, token=tok).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    crs_tab = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site")
    E = np.zeros((len(F), 2 * model.config.hidden_size), np.float32); ok = np.zeros(len(F), bool)
    for region, idx in F.groupby("region").groups.items():
        idx = np.asarray(list(idx)); todo = np.ones(len(idx), bool)
        for path in sorted(glob.glob(os.path.join(DATA, region, "04_*", "*.tif"))):
            with rasterio.open(path) as ds:
                crs = f"EPSG:{int(crs_tab.loc[region, 'json_epsg'])}" if region in crs_tab.index else ds.crs.to_string()
                X, Y = Transformer.from_crs(3857, crs, always_xy=True).transform(F.x.values[idx], F.y.values[idx])
                a = ds.read(indexes=list(range(1, min(3, ds.count) + 1))).astype(np.float32); T = ds.transform
            res = abs(T.a)
            chips, where = [], []
            for j in np.flatnonzero(todo):
                c = chip(a, T, res, X[j], Y[j])
                if c is not None:
                    chips.append(c); where.append(j)
            for i in range(0, len(chips), 64):
                b = proc(images=chips[i:i + 64], return_tensors="pt").to(dev)
                if dev == "cuda":
                    b = {k: v.half() for k, v in b.items()}
                with torch.no_grad():
                    h = model(**b).last_hidden_state.float()
                emb = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy()
                for k, j in enumerate(where[i:i + 64]):
                    E[idx[j]] = emb[k]; ok[idx[j]] = True; todo[j] = False
        print(f"dinov3 {region}: {int(ok[idx].sum())}/{len(idx)} candidates embedded", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True); np.savez(cache, E=E, ok=ok)
    return E, ok


def main():
    import geopandas as gpd
    from shapely.geometry import Point
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls, K
    from robust_accept import site_counts, report
    M = models(extra=list(NAMES[3:])); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    RF = region_folds(); folds = sorted({RF[r] for r in regions})
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    Sg = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); geom = scene_geometry()
    F0 = fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    GC = F0[(F0.cls == "gorodishcha") & (F0.score >= 0.02)]
    F0 = F0[F0.score >= 0.10].reset_index(drop=True)
    E, ok = embeddings(F0, os.path.join(ROOT, "work/spor/oof_dinov3.npz"))
    fam = lambda c: "k" if c in K else c
    y = np.zeros(len(F0), int)
    for r, g in F0.groupby("region"):
        lab = L[L.region == r]
        for i, row in g.iterrows():
            p = Point(row.x, row.y)
            y[i] = int(any(fam(c) == fam(row.cls) and geo.contains(p) for geo, c in zip(lab.geometry, lab.class_name)))
    v = np.full(len(F0), np.nan)
    fold = F0.region.map(RF).values
    for f in folds:
        tr = (fold != f) & ok; te = (fold == f) & ok
        sc = StandardScaler().fit(E[tr])
        m = LogisticRegression(max_iter=3000, C=0.3, class_weight="balanced").fit(sc.transform(E[tr]), y[tr])
        v[te] = m.predict_proba(sc.transform(E[te]))[:, 1]
    v = np.where(np.isnan(v), F0.score.values, v)                 # no chip (scene edge): keep the detector score
    kur = F0.cls.isin(K).values
    print(f"{len(regions)} regions, {len(F0)} candidates (score ≥ 0.10), {int(ok.sum())} embedded, {int(y.sum())} inside GT", flush=True)
    from sklearn.metrics import roc_auc_score
    print(f"kurgan candidates OOF AUC: detector score {roc_auc_score(y[kur], F0.score[kur]):.3f} | verifier {roc_auc_score(y[kur], v[kur]):.3f} | "
          f"0.5/0.5 blend {roc_auc_score(y[kur], 0.5 * F0.score[kur] + 0.5 * v[kur]):.3f}")
    half = {}
    for r in Sg.itertuples():
        g = geom.get(r.region, [])
        half[r.Index] = min(g, key=lambda s: np.hypot(r.x - s[0], r.y3857 - s[1]))[2] if g else 300.0

    def pipeline(lam, regs):
        F = F0.copy()
        F.loc[kur, "score"] = (1 - lam) * F.score[kur] + lam * v[kur]
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
            c = GC[(GC.region == r.region) & (np.hypot(GC.x - cx, GC.y - cy) <= 0.2 * half[r.Index])]
            if len(c):
                b = c.loc[c.score.idxmax()]; cx, cy = b.x, b.y
            rows.append((r.region, "gorodishcha", r.p, cx, cy))
        return pd.concat([G, pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)

    gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    C = fold_counts(lambda lam, f: fc(pipeline(lam, [r for r in regions if RF[r] == f])), LAMBDAS, folds, gt)
    for lam in LAMBDAS:
        tot = _add_all(C[lam].values())
        print(f"λ={lam:<4} pooled {weighted(tot):.4f}  folds [{' '.join(f'{weighted(C[lam][f]):.4f}' for f in folds)}]", flush=True)
    h, ch, o, os_, _ = nested(C); base = weighted(_add_all(C[0.0].values()))
    print(f"deployed (λ=0) {base:.4f} | nested honest {h:.4f} ({h - base:+.4f}) | oracle {o:.4f} @ λ={os_} | chosen {ch}")
    print(f"worst outer-fold change {min(weighted(C[ch[f]][f]) - weighted(C[0.0][f]) for f in folds):+.4f}")
    lam = max(set(ch.values()), key=list(ch.values()).count)
    gtr = {r: to_gt_geojson(L[L.region == r]) for r in regions}
    CS = site_counts(lambda s, r: fc(pipeline(s, [r])), (0.0, lam), regions, gtr)
    report(f"dinov3 verifier λ={lam}", CS[0.0], CS[lam])


if __name__ == "__main__":
    main()
