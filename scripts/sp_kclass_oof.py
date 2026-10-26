"""Satellite damaged / intact crop classifier (two-stage kurgans), honest on the shared 4 site folds:
ConvNeXt-T (ImageNet) on 96×96 px (48 m @ 0.5 m) crops around labelled kurgans, trained on the other folds; applied to
every fused kurgan detection of the v12 pipeline (6 members, arch 0.15, framing prior, hillfort gate) to (a) assign the
class by p_intact and (b) hedge when p_intact ∈ [lo, hi]. Nested vs the deployed member-vote class + hedge 0.4.
  python scripts/sp_kclass_oof.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")
SP = os.path.join(ROOT, "kaggle_import/code/work/sp")
CR = 96
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None] * 255
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None] * 255


def scene_index():
    from rasterio.transform import Affine
    from pyproj import Transformer
    S = pd.read_csv(os.path.join(SP, "scenes.csv")); crs = pd.read_csv(os.path.join(ROOT, "kaggle_import/code/docs/site_crs.csv")).set_index("site")
    out = {}
    for r in S.itertuples():
        T = Affine(*json.loads(r.T)); c = f"EPSG:{int(crs.loc[r.site, 'json_epsg'])}" if r.site in crs.index else r.crs
        out.setdefault(r.site, []).append((r.scene, T, Transformer.from_crs(3857, c, always_xy=True), r.W, r.H))
    return out


_IMG = {}


def crop(idx, region, x, y):
    for sc, T, tr, W, H in idx.get(region, []):
        X, Y = tr.transform(x, y); c, rr = ~T * (X, Y)
        if 0 <= c < W and 0 <= rr < H:
            if sc not in _IMG:
                _IMG[sc] = np.load(os.path.join(SP, "img", sc + ".npy"), mmap_mode="r")
            a = _IMG[sc]; c0, r0 = int(c) - CR // 2, int(rr) - CR // 2
            out = np.zeros((3, CR, CR), np.float32)
            ys, xs = slice(max(0, r0), min(H, r0 + CR)), slice(max(0, c0), min(W, c0 + CR))
            if ys.start < ys.stop and xs.start < xs.stop:
                out[:, ys.start - r0:ys.stop - r0, xs.start - c0:xs.stop - c0] = a[:3, ys, xs]
            return out
    return None


def main():
    import geopandas as gpd
    import timm
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc
    from sp_fusion4_oof import gate
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls
    idx = scene_index(); RF = region_folds()
    Lall = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); Lk = Lall[(Lall["mod"] == "SpOr") & Lall.class_name.isin(K)]
    rows = []
    for g in Lk.itertuples():
        c = g.geometry.representative_point(); im = crop(idx, g.region, c.x, c.y)
        if im is not None and g.region in RF:
            rows.append((g.region, RF[g.region], int(g.class_name == K[1]), im))
    print(len(rows), "labelled kurgan crops; intact", sum(r[2] for r in rows))
    folds = sorted({r[1] for r in rows}); models_ = {}
    rng = np.random.default_rng(0)
    for f in folds:
        tr = [r for r in rows if r[1] != f]
        y = np.array([r[2] for r in tr]); w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1)); w /= w.sum()
        m = timm.create_model("convnext_tiny", pretrained=True, num_classes=1).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
        for it in range(1500):
            b = rng.choice(len(tr), 32, p=w); X = []
            for i in b:
                im = tr[i][3].copy()
                im = np.rot90(im, rng.integers(4), axes=(1, 2)); im = im[:, :, ::-1] if rng.random() < 0.5 else im
                im = im * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
                X.append((np.clip(im, 0, 255) - MEAN) / STD)
            X = torch.from_numpy(np.ascontiguousarray(np.stack(X), np.float32)).cuda(); t = torch.tensor([tr[i][2] for i in b], dtype=torch.float32).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(X).float().squeeze(1), t)
            opt.zero_grad(); loss.backward(); opt.step()
        models_[f] = m.eval(); print("fold", f, "trained", flush=True)

    @torch.no_grad()
    def p_intact(f, ims):
        X = torch.from_numpy(np.stack([(im - MEAN) / STD for im in ims]).astype(np.float32)).cuda()
        ps = [torch.sigmoid(models_[f](X.flip(fl) if fl else X).float()).squeeze(1) for fl in ((), (-1,), (-2,))]
        return torch.stack(ps).mean(0).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    oof_y, oof_p = [], []
    for f in folds:
        te = [r for r in rows if r[1] == f]
        if te:
            oof_p += list(p_intact(f, [r[3] for r in te])); oof_y += [r[2] for r in te]
    print(f"damaged/intact crop classifier OOF AUC {roc_auc_score(oof_y, oof_p):.3f}")
    # apply to the v12 pipeline's fused kurgan detections
    names = ("dfine", "rtdetr", "hm", "dfnodn", "balq", "balq3")
    M = models(extra=["dfnodn", "balq", "balq3"]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; folds2 = sorted({RF[r] for r in regions})
    L = Lall[(Lall["mod"] == "SpOr") & Lall.region.isin(regions)]; gt = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds2}
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")); S = S[S.region.isin(regions)]
    F = fuse_cls([M[n] for n in names]); F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)].reset_index(drop=True)
    geom = scene_geometry(); keep = []
    for r in F.itertuples():
        d = min([(np.hypot(r.x - cx, r.y - cy) / max(h, 1), sm) for cx, cy, h, sm in geom.get(r.region, [])] or [(0.0, True)])
        keep.append(d[1] or d[0] <= 0.5 or r.score >= 0.35)
    F = F[np.array(keep)].reset_index(drop=True)
    F["pint"] = np.nan
    for f in folds2:
        sel = F[(F.region.map(RF) == f) & F.cls.isin(K)]
        ims = [crop(idx, r.region, r.x, r.y) for r in sel.itertuples()]; ok = [i for i, im in enumerate(ims) if im is not None]
        if ok:
            F.loc[sel.index[ok], "pint"] = p_intact(f, [ims[i] for i in ok])

    def make(s, f):
        mode, a, b = s
        G = F[F.region.map(RF) == f].copy(); ex = []
        for r in G[G.cls.isin(K)].itertuples():
            if mode == "vote":                                   # deployed: member class + disagreement hedge 0.4
                o = K[1] if r.cls == K[0] else K[0]
                if (r.s_kt if o == K[1] else r.s_kp) >= 0.4:
                    ex.append((r.region, o, r.score * 0.99, r.x, r.y))
            elif not np.isnan(r.pint):
                primary = K[1] if r.pint >= a else K[0]
                G.loc[r.Index, "cls"] = primary
                if b > 0 and abs(r.pint - a) < b:                # hedge only when the classifier is uncertain
                    ex.append((r.region, K[0] if primary == K[1] else K[1], r.score * 0.99, r.x, r.y))
        G = pd.concat([G[["region", "cls", "score", "x", "y"]], pd.DataFrame(ex, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
        return fc(gate(G, S[S.region.map(RF) == f]))
    SS = [("vote", 0, 0)] + [("clf", a, b) for a in (0.3, 0.4, 0.5, 0.6) for b in (0.0, 0.15, 0.3)]
    C = fold_counts(make, SS, folds2, gt)
    for s in SS:
        tot = _add_all(C[s].values())
        print(f"{s}: pooled {weighted(tot):.4f}  kp {tuple(int(v) for v in tot[K[0]])}  kt {tuple(int(v) for v in tot[K[1]])}")
    h, ch, o, os_, _ = nested(C)
    print(f"deployed vote+hedge {weighted(_add_all(C[('vote', 0, 0)].values())):.4f} | nested honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {list(ch.values())}")


if __name__ == "__main__":
    main()
