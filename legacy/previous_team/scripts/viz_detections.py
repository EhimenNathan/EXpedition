"""Detection figures for ablations and presentations: one held-out scene, one panel per run, coloured by the OFFICIAL
match (koz3-1.5.0 _match_polygon_gt, greedy one-to-one by confidence, per class) so pictures agree with the scores.

    python scripts/viz_detections.py --runs "U-Net R34=work/sp_runs/sp_r34_g:deploy" "D-FINE-L=work/sp_runs/sp_dfine_l:0.5" \
        --auto 6 [--scenes 163_Курганы_34__0,...] [--data data_sp] [--out docs/figures/detections] [--tag sp_unet_vs_dfine]

Run spec "<label>=<OOF dir>:<policy>": policy "deploy" uses <dir>/deploy_thresholds.json (per class mask + conf),
a number uses that confidence for every class (single-threshold candidates, thr column 0).
Green = true positive, red = false alarm, yellow outline = missed label. Letters: T intact kurgan, D damaged kurgan,
H hillfort, F fortification, A architecture. Scenes come from the OOF folds, so every panel is out-of-fold.
Images: <out>/<tag>__<scene>.png plus a contact sheet <out>/<tag>__sheet.png."""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from shapely import wkb
from shapely.geometry import box
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from koz3 import sp_io
from koz3.official import compute_metrics as CM

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--scenes", default=""); ap.add_argument("--auto", type=int, default=6)
ap.add_argument("--data", default=os.path.join(ROOT, "data_sp")); ap.add_argument("--gsd", type=float, default=0.5)
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "detections")); ap.add_argument("--tag", default="det")
ap.add_argument("--margin_m", type=float, default=40)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CALC = CM.F1Calculator(input_crs_mode="utm")
LET = {"kurgany_tselye": "T", "kurgany_povrezhdennye": "D", "gorodishcha": "H", "fortifikatsii": "F", "arkhitektury": "A"}

runs = []
for spec in a.runs:
    label, rest = spec.split("=", 1); d, pol = rest.rsplit(":", 1)
    P = pd.read_parquet(f"{d}/oof_pred.parquet"); P["lcls"] = P.cls.map(CM.normalize_class_name)
    if pol == "deploy":
        th = json.load(open(f"{d}/deploy_thresholds.json", encoding="utf-8"))
        P = pd.concat([P[(P.lcls == c) & (np.isclose(P.thr, v["mask"])) & (P.score >= v["conf"])] for c, v in th.items()])
    else:
        P = P[P.score >= float(pol)]
    runs.append((label, P))
G = pd.read_parquet(f"{a.runs[0].split('=', 1)[1].rsplit(':', 1)[0]}/oof_gt.parquet"); G["lcls"] = G.cls.map(CM.normalize_class_name)
scenes = a.scenes.split(",") if a.scenes else list(G.groupby("scene").size().sort_values(ascending=False).index[:a.auto])


def match(preds, gts):
    """preds [(geom, score, cls)], gts [(geom, cls)] -> (tp pred idx set, matched gt idx set); official rule per class."""
    tp, used = set(), set()
    for i in sorted(range(len(preds)), key=lambda i: -preds[i][1]):
        best, bj = -1e9, -1
        for j, (g, c) in enumerate(gts):
            if j in used or c != preds[i][2]:
                continue
            hit, s = CALC._match_polygon_gt(preds[i][0], g)
            if hit and s > best:
                best, bj = s, j
        if bj >= 0:
            tp.add(i); used.add(bj)
    return tp, used


def scene_image(scene):
    site, k = scene.rsplit("__", 1)
    tifs = sorted(glob.glob(os.path.join(a.data, site, "04_*", "*.tif")))
    return sp_io.read_scene(tifs[int(k.split("_")[0])], a.gsd)


def draw(ax, img, T, gts, preds, title, view):
    x0, y0, x1, y1 = view
    ax.imshow(np.moveaxis(img[:3], 0, -1), extent=(T.c, T.c + T.a * img.shape[2], T.f + T.e * img.shape[1], T.f))
    tp, used = match(preds, gts)
    for j, (g, c) in enumerate(gts):
        if j not in used:
            ax.add_patch(MPoly(np.asarray(g.exterior.coords), closed=True, fill=False, ec="#ffd400", lw=1.6))
    for i, (g, s, c) in enumerate(preds):
        col = "#22c55e" if i in tp else "#ef4444"
        ax.add_patch(MPoly(np.asarray(g.exterior.coords), closed=True, fc=col, ec=col, alpha=0.35, lw=1.2))
        p = g.representative_point(); ax.text(p.x, p.y, LET.get(c, "?"), color="white", fontsize=6, ha="center", va="center", weight="bold")
    n_fp = len(preds) - len(tp); n_fn = len(gts) - len(used)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nTP {len(tp)}  FP {n_fp}  missed {n_fn}", fontsize=9)
    return len(tp), n_fp, n_fn


paths = []
for sc in scenes:
    try:
        img, T, crs, _ = scene_image(sc)
    except Exception as e:
        print("skip", sc, repr(e)); continue
    gts = [(wkb.loads(x), c) for x, c in zip(G[G.scene == sc].wkb, G[G.scene == sc].lcls)]
    per = [[(wkb.loads(x), s, c) for x, s, c in zip(P[P.scene == sc].wkb, P[P.scene == sc].score, P[P.scene == sc].lcls)] for _, P in runs]
    geoms = [g for g, _ in gts] + [g for pr in per for g, _, _ in pr]
    if not geoms:
        continue
    bx = box(*geoms[0].bounds)
    for g in geoms[1:]:
        bx = bx.union(box(*g.bounds))
    l, b, r, t = bx.bounds; m = a.margin_m; side = max(r - l, t - b) + 2 * m; cx, cy = (l + r) / 2, (b + t) / 2
    view = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.6), squeeze=False)
    for ax, (label, _), pr in zip(axs[0], runs, per):
        draw(ax, img, T, gts, pr, label, view)
    fig.suptitle(f"{sc}  (out-of-fold; green TP, red FP, yellow = missed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); p = os.path.join(a.out, f"{a.tag}__{sc}.png"); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    print("saved", p, flush=True)
if paths:  # contact sheet: one row per scene
    ims = [plt.imread(p) for p in paths]; h = sum(i.shape[0] for i in ims); w = max(i.shape[1] for i in ims)
    sheet = np.ones((h, w, ims[0].shape[2]), ims[0].dtype); y = 0
    for i in ims:
        sheet[y:y + i.shape[0], :i.shape[1]] = i; y += i.shape[0]
    plt.imsave(os.path.join(a.out, f"{a.tag}__sheet.png"), sheet); print("sheet", os.path.join(a.out, f"{a.tag}__sheet.png"))
