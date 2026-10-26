"""Satellite D-FINE (SpOr-trained) on Or chips resampled 0.1 → 0.5 m; held-out fold model per site; nested vs Or labels."""
import os, sys
import numpy as np, pandas as pd, cv2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    from pyproj import Transformer
    from rasterio.transform import Affine
    import geopandas as gpd
    from koz3.det_train import load_eager, detect_chip
    from koz3.sp_train import load_chips
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    from sp_assemble_oof import region_folds
    from sp_fusion_oof import fc
    PREV = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
    RF = region_folds(); chips = load_chips("Or"); models = {}; rows = []
    for c in chips:
        f = RF.get(c["region"], 0)
        if f not in models:
            models[f] = load_eager("ustc-community/dfine-large-coco", os.path.join(ROOT, f"kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/fold{f}.pt"))
        rgb = cv2.imread(os.path.join(ROOT, "data/or_prep/chips", c["file"]))[:, :, ::-1]
        small = cv2.resize(rgb, (max(1, rgb.shape[1] // 5), max(1, rgb.shape[0] // 5)), interpolation=cv2.INTER_AREA)
        t = Affine(*c["transform"]) * Affine.scale(5, 5); tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
        for cl, cx, cy, bw, bh, s in detect_chip(models[f], small, tile=640, ov=160):
            X, Y = tr.transform(*(t * (cx, cy)))
            rows.append((c["region"], PREV[cl], s, X, Y))
    D = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"]); print(len(D), "candidates", D.cls.value_counts().to_dict())
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "Or"]
    regs = sorted({c["region"] for c in chips}); fold = {r: i for i, r in enumerate(regs)}
    gt = {i: to_gt_geojson(L[L.region == r]) for r, i in fold.items()}
    S = [(t, m) for t in (0.3, 0.5, 0.7) for m in ("all", "arch")]
    mk = lambda s, f: fc(D[(D.score >= s[0]) & (D.region.map(fold) == f) & ((D.cls == "arkhitektury") if s[1] == "arch" else True)])
    C = fold_counts(mk, S, sorted(gt), gt)
    h, ch, o, os_, tot = nested(C)
    print(f"SpOr D-FINE on Or: honest {h:.4f} oracle {o:.4f} @ {os_}"); print({k: v for k, v in tot.items() if v[0] + v[2]})


if __name__ == "__main__":
    main()
