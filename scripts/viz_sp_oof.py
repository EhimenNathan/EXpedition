"""figures/detections/sp_oof_*.png — satellite OOF on sample scenes: labels (outlines by class), fused detector
peaks (dots, size ∝ score) and hillfort-localiser points (stars). Everything out-of-fold."""
import json
import os
import sys

import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
COL = {"kurgany_povrezhdennye": "#e8590c", "kurgany_tselye": "#fcc419", "gorodishcha": "#e64980", "arkhitektury": "#4dabf7", "fortifikatsii": "#94d82d"}


def main(n_small=8, n_big=4, det_thr=0.3):
    from sp_assemble_oof import models
    from sp_fusion_oof import fuse
    ann = json.load(open(os.path.join(ROOT, "data/spor_prep/ann.json")))
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]
    M = models(); Fz = fuse(list(M.values()), np.ones(len(M))); Fz = Fz[Fz.score >= det_thr]
    G = pd.read_parquet(os.path.join(ROOT, "work/spor/oof_gor_cands.parquet")); G = G[(G.thr == 0.5) & (G.score >= 0.5)]
    rng = np.random.default_rng(1)
    small = [c for c in ann if c["kind"] == "small" and c["region"] in set(Fz.region)]
    big = [c for c in ann if c["kind"] == "tile" and len(c["boxes"]) >= 5 and c["region"] in set(Fz.region)]
    pick = list(rng.choice(small, min(n_small, len(small)), replace=False)) + list(rng.choice(big, min(n_big, len(big)), replace=False))
    os.makedirs(os.path.join(ROOT, "figures/detections"), exist_ok=True)
    for c in pick:
        img = cv2.imread(os.path.join(ROOT, "data/spor_prep/chips", c["file"]))[:, :, ::-1]
        inv = ~Affine(*c["transform"]); tr = Transformer.from_crs(3857, c["crs"], always_xy=True)
        topx = lambda X, Y: inv * tr.transform(X, Y)
        fig, ax = plt.subplots(figsize=(8, 8 * img.shape[0] / img.shape[1]))
        ax.imshow(img)
        lab = L[L.region == c["region"]].to_crs(c["crs"])
        for g, cl in zip(lab.geometry, lab.class_name):
            for p in getattr(g, "geoms", [g]):
                xs, ys = zip(*[inv * xy for xy in p.exterior.coords])
                ax.plot(xs, ys, color=COL[cl], lw=1.2)
        for r in Fz[Fz.region == c["region"]].itertuples():
            px, py = topx(r.x, r.y)
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                ax.scatter(px, py, s=20 + 80 * r.score, c=COL[r.cls], edgecolors="white", linewidths=0.6)
        for r in G[G.region == c["region"]].itertuples():
            px, py = topx(r.x, r.y)
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                ax.scatter(px, py, s=160, marker="*", c=COL["gorodishcha"], edgecolors="white")
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.axis("off")
        ax.set_title(f"{c['region']} ({c['kind']}) — lines: labels, dots: fused detections ≥{det_thr}, ★ hillfort localiser", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/detections", f"sp_oof_{c['file'][:-4]}.png"), dpi=80); plt.close(fig)
    print("saved", len(pick))


if __name__ == "__main__":
    main()
