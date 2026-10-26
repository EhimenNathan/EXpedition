"""Hillfort localisation inside gated satellite small scenes with foundation patch features: frozen DINOv2 ViT-B/14
patch tokens (27×27 per scene, identity view) → linear probe "patch centre inside a hillfort" trained on the OTHER folds
only → smoothed per-scene map → its peak replaces the scene centre as the emitted hillfort point. Compared, nested, in
the v12 satellite pipeline (5-member fusion, architecture 0.15, hillfort gate "swap" g 0.1).

  python scripts/sp_gor_localise_oof.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
G = 27; S_IN = 378


@torch.no_grad()
def patch_tokens(chips):
    import timm
    import torch.nn.functional as F
    from koz3.scene_cls import load
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=S_IN).cuda().eval()
    out = []
    for c in chips:
        x = F.interpolate(torch.from_numpy(load(c))[None].cuda(), size=(S_IN, S_IN), mode="bilinear", align_corners=False)
        t = m.forward_features(x)[0, 1:]                       # (729, 768)
        out.append(t.float().cpu().numpy().astype(np.float16))
    return np.stack(out)


def patch_geo(c):
    """3857 coordinates of the 27×27 patch centres of a chip (inverse of scene_cls.load: long side → 384, centred pad)."""
    from pyproj import Transformer
    from rasterio.transform import Affine
    w, h = c["w"], c["h"]; f = 384 / max(w, h)
    ow, oh = round(w * f), round(h * f); ox, oy = (384 - ow) // 2, (384 - oh) // 2
    u = (np.arange(G) + 0.5) * (S_IN / G) * (384 / S_IN)               # patch centre in 384 coords
    U, V = np.meshgrid(u, u)
    px, py = (U - ox) / f, (V - oy) / f                               # chip pixels
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    t = Affine(*c["transform"]); X, Y = t * (px, py)
    mx, my = Transformer.from_crs(c["crs"], 3857, always_xy=True).transform(X, Y)
    return np.asarray(mx), np.asarray(my), inside


def main():
    import geopandas as gpd
    from scipy.ndimage import gaussian_filter
    from shapely import STRtree
    from shapely.geometry import Point
    from sklearn.linear_model import LogisticRegression
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested, weighted, _add_all
    from koz3.scene_cls import scenes
    from sp_assemble_oof import models, region_folds
    from sp_fusion_oof import fc, fuse
    chips = scenes()
    cache = os.path.join(ROOT, "work/spor/dinov2b_patch_tokens.npy")
    T = np.load(cache) if os.path.exists(cache) else patch_tokens(chips)
    if not os.path.exists(cache):
        np.save(cache, T)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet"))
    Lg = L[(L["mod"] == "SpOr") & (L.class_name == "gorodishcha")]
    geoms = list(Lg.geometry); tree = STRtree(geoms)
    geo = [patch_geo(c) for c in chips]
    Y = np.zeros((len(chips), G, G), np.int8)
    for i, (mx, my, ins) in enumerate(geo):
        for a in range(G):
            for b in range(G):
                if ins[a, b]:
                    pt = Point(mx[a, b], my[a, b])
                    Y[i, a, b] = any(geoms[j].contains(pt) for j in tree.query(pt))
    fold = np.array([c["fold"] for c in chips])
    print(len(chips), "scenes;", int(Y.sum()), "hillfort patches in", int((Y.reshape(len(chips), -1).max(1)).sum()), "scenes")
    heat = np.zeros((len(chips), G, G), np.float32)
    for f in sorted(set(fold)):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        Xtr = T[tr].reshape(-1, T.shape[-1]).astype(np.float32); ytr = Y[tr].reshape(-1)
        mask = np.concatenate([geo[i][2].reshape(-1) for i in tr])
        mu, sd = Xtr[mask].mean(0), Xtr[mask].std(0) + 1e-6
        clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((Xtr[mask] - mu) / sd, ytr[mask])
        for i in te:
            p = clf.predict_proba((T[i].astype(np.float32) - mu) / sd)[:, 1].reshape(G, G)
            heat[i] = np.where(geo[i][2], gaussian_filter(p, 1.5), -1)
    rows = []
    for i, c in enumerate(chips):
        a, b = np.unravel_index(np.argmax(heat[i]), heat[i].shape)
        mx, my, _ = geo[i]
        pk = Point(mx[a, b], my[a, b]); cc = Point(mx[G // 2, G // 2], my[G // 2, G // 2])
        hit = lambda pt: any(geoms[j].contains(pt) for j in tree.query(pt))
        rows.append((c["file"], c["region"], c["y"], float(mx[a, b]), float(my[a, b]), hit(pk), hit(cc)))
    D = pd.DataFrame(rows, columns=["file", "region", "y", "px", "py", "hit_peak", "hit_centre"])
    pos = D[D.y == 1]
    print(f"hillfort scenes: point inside the hillfort — centre {pos.hit_centre.mean():.2f}  foundation peak {pos.hit_peak.mean():.2f}  (n={len(pos)})")
    # full nested study in the v12 satellite pipeline
    S = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_scene_cls.parquet")).merge(D[["file", "px", "py"]], on="file")
    M = models(extra=["dfnodn", "balq"]); regions = sorted(set.intersection(*[set(m.region) for m in M.values()]))
    M = {k: m[m.region.isin(regions)] for k, m in M.items()}; RF = region_folds(); folds = sorted({RF[r] for r in regions})
    Lr = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt = {f: to_gt_geojson(Lr[Lr.region.map(RF) == f]) for f in folds}
    S = S[S.region.isin(regions)]
    F = fuse([M[n] for n in ("dfine", "rtdetr", "hm", "dfnodn", "balq")], np.ones(5))
    F = F[F.score >= np.where(F.cls == "arkhitektury", 0.15, 0.25)]

    def make(s, f):
        mode = s[0]; g = 0.1
        G_ = F[F.region.map(RF) == f]; c = S[S.region.map(RF) == f]
        G_ = G_[~((G_.cls == "gorodishcha") & G_.region.isin(set(c[c.p < g].region)))]
        a = c[c.p >= g]
        pts = [pd.DataFrame({"region": a.region, "cls": "gorodishcha", "score": a.p, "x": a.x, "y": a.y3857})]
        if mode in ("peak", "both"):
            pts.append(pd.DataFrame({"region": a.region, "cls": "gorodishcha", "score": a.p * (0.99 if mode == "both" else 1), "x": a.px, "y": a.py}))
            if mode == "peak":
                pts = pts[1:]
        return fc(pd.concat([G_] + pts, ignore_index=True))
    SS = [("centre",), ("peak",), ("both",)]
    C = fold_counts(make, SS, folds, gt)
    for s in SS:
        tot = _add_all(C[s].values())
        print(f"{s[0]:7s} pooled {weighted(tot):.4f}  gor tp/fp/fn {tuple(int(x) for x in tot['gorodishcha'])}")
    h, ch, o, os_, _ = nested(C)
    print(f"nested honest {h:.4f} chosen {ch}")


if __name__ == "__main__":
    main()
