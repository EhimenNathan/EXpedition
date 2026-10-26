"""Figures of detections under the official matching rule: true positives (green), false alarms (red), misses (yellow).

    python scripts/fig_detections.py lidar     # site 180 (only local DTM), A2 out-of-fold candidates, deployed thresholds
    python scripts/fig_detections.py finds     # geophysics finds, leave-one-region-out LightGBM (work/geo/finds_candidates.parquet)
    python scripts/fig_detections.py objects   # geophysics objects, leave-one-region-out probability maps (work/geo/objects_oof)
All predictions shown come from models that never saw the region/site (out-of-fold). Output: docs/figures/det_*.png"""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3.official.fastmatch import soft_match, prf

C_TP, C_FP, C_FN = "#22c55e", "#ef4444", "#facc15"


def outline(ax, g, color, lw=1.6, ls="-"):
    for p in (g.geoms if g.geom_type.startswith("Multi") else [g]):
        x, y = p.exterior.xy; ax.plot(x, y, color=color, lw=lw, ls=ls)


def lidar():
    import rasterio
    from koz3 import dtm as D
    site = "180_Гнёздово_Лесная_2"
    sys.path.insert(0, "kaggle")
    from official_rescore import load
    P, G = load("A2_ref_unet_r34_all", groups=[site], thr=0.5)
    thr = json.load(open("kaggle/out/lidar_conf_thresholds.json"))
    with rasterio.open("work/tdata/lidar/180_Гнёздово_Лесная_2/dtm.tif") as r:
        z = r.read(1).astype(np.float32); T = r.transform; gsd = abs(T.a)
    v = z > -9000; z[~v] = np.median(z[v])
    hs = np.mean([D.hillshade(z, az, 35, gsd) for az in (315, 45, 135, 225)], 0); lrm = D.lrm(z, 7.5, gsd)
    ext = (T.c, T.c + T.a * z.shape[1], T.f + T.e * z.shape[0], T.f)
    names = {"kurgany_tselye": "intact kurgans", "kurgany_povrezhdennye": "damaged kurgans"}
    res = {}
    for cls in names:
        pc = P[(P.lcls == cls) & (P.score >= thr[cls])]; gc = list(G[G.lcls == cls].geometry)
        tp, fp, fn, pairs = soft_match(list(zip(pc.geometry, pc.score)), gc)
        mp = {id(p) for p, _ in pairs}; mg = {id(g) for _, g in pairs}
        res[cls] = dict(tp=[(p, g) for p, g in pairs], fp=[p for p in pc.geometry if id(p) not in mp], fn=[g for g in gc if id(g) not in mg],
                        prf=prf(tp, fp, fn))
    fig = plt.figure(figsize=(20, 13)); gs = fig.add_gridspec(3, 6)
    for k, cls in enumerate(names):
        ax = fig.add_subplot(gs[:, k * 2:(k + 1) * 2] if False else gs[:2, k * 3:(k + 1) * 3])
        ax.imshow(hs, cmap="gray", extent=ext); ax.imshow(np.clip(lrm, -0.5, 0.5), cmap="RdBu_r", alpha=0.25, extent=ext)
        R = res[cls]
        for p, g in R["tp"]: outline(ax, p, C_TP, 1.2)
        for p in R["fp"]: outline(ax, p, C_FP, 1.0)
        for g in R["fn"]: outline(ax, g, C_FN, 1.0, "--")
        P_, R_, F_ = R["prf"]
        ax.set_title(f"{names[cls]}: TP {len(R['tp'])} (green), FP {len(R['fp'])} (red), missed {len(R['fn'])} (yellow dashed)\n"
                     f"official P {P_:.2f}  R {R_:.2f}  F1 {F_:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    # zoomed true positives: 3 per class, largest-confidence first
    j = 0
    for cls in names:
        tps = sorted(res[cls]["tp"], key=lambda pg: -pg[0].area)[:3]
        for p, g in tps:
            ax = fig.add_subplot(gs[2, j]); j += 1
            c = g.centroid; w = max(12, 1.6 * np.sqrt(g.area))
            ax.imshow(hs, cmap="gray", extent=ext); ax.imshow(np.clip(lrm, -0.5, 0.5), cmap="RdBu_r", alpha=0.3, extent=ext)
            outline(ax, g, "white", 2.0, "--"); outline(ax, p, C_TP, 2.0)
            ax.set_xlim(c.x - w, c.x + w); ax.set_ylim(c.y - w, c.y + w); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{names[cls]} (label dashed white)\narea {g.area:.0f} m2", fontsize=9)
    fig.suptitle("LiDAR expert A2 on site 180 (Гнёздово, forest), out-of-fold: the fold model never saw this site. "
                 "Background: multi-direction hillshade + local relief (red = higher).", fontsize=12)
    plt.tight_layout(); plt.savefig("docs/figures/det_lidar_180_kurgans.png", dpi=95); print("saved det_lidar_180_kurgans.png")


def finds():
    from koz3 import geo_data as GD
    C = pd.read_parquet("work/geo/finds_candidates.parquet")
    U = {f"{u['region']}|{u['key']}": u for u in GD.discover("data_geo") if u["method"] == "MagGr" and u["finds"]}
    show = [uid for uid in ["204_Семибратнее_Городище_4|", "205_Семибратнее_Городище_5|", "209_Гнёздово_Терраса_1|", "212_Гнёздово_Усадьба|цвет_2"] if uid in set(C.uid)]
    fig, axs = plt.subplots(2, len(show), figsize=(6 * len(show), 12), squeeze=False)
    for k, uid in enumerate(show):
        d = GD.load_unit(U[uid], 0.1); g = C[(C.uid == uid) & (C.p >= 0.2)]
        gts = d["finds"]; preds = [(Point(x, y).buffer(0.25, 8), p) for x, y, p in zip(g.x, g.y, g.p)]
        tp, fp, fn, pairs = soft_match(preds, gts)
        mp = {id(p) for p, _ in pairs}; mg = {id(q) for _, q in pairs}
        T = d["T"]; img = d["lum"]; ext = (T.c, T.c + T.a * img.shape[1], T.f + T.e * img.shape[0], T.f)
        for row in range(2):
            ax = axs[row, k]; ax.imshow(np.where(d["valid"], img, np.nan), cmap="gray", extent=ext)
            xs = [p.centroid.x for p, _ in preds]; ys = [p.centroid.y for p, _ in preds]
            col = [C_TP if id(p) in mp else C_FP for p, _ in preds]
            ax.scatter(xs, ys, s=18 if row else 6, facecolors="none", edgecolors=col, linewidths=1.2)
            miss = [q for q in gts if id(q) not in mg]
            ax.scatter([q.x for q in miss], [q.y for q in miss], s=18 if row else 6, marker="x", c=C_FN)
            if row:  # zoom on the densest true-positive area
                tpx = np.array([[p.centroid.x, p.centroid.y] for p, _ in pairs])
                if len(tpx):
                    cx, cy = np.median(tpx, 0); ax.set_xlim(cx - 12, cx + 12); ax.set_ylim(cy - 12, cy + 12)
            ax.set_xticks([]); ax.set_yticks([])
        P_, R_, F_ = prf(tp, fp, fn)
        axs[0, k].set_title(f"{uid.split('|')[0]}\nTP {tp}, FP {fp}, missed {fn}; official F1 {F_:.2f}", fontsize=10)
        axs[1, k].set_title("zoom 24 m x 24 m", fontsize=9)
    fig.suptitle("Geophysics point finds (magnetometry), leave-one-region-out LightGBM, p >= 0.2: green circle = found (centroid within 2 m), "
                 "red circle = false alarm, yellow x = missed", fontsize=12)
    plt.tight_layout(); plt.savefig("docs/figures/det_geo_finds.png", dpi=90); print("saved det_geo_finds.png")


def objects():
    import rasterio.features as RF
    from koz3 import geo_data as GD
    sys.path.insert(0, "scripts")
    from scipy import ndimage as ndi
    from shapely.geometry import shape
    U = {f"{u['region']}|{u['method']}|{u['key']}": u for u in GD.discover("data_geo") if u["objects"]}
    GSD = {"MagGr": 0.2, "EtGr": 0.1, "TerFl": 0.3}
    picks = ["209_Гнёздово_Терраса_1|MagGr|", "215_Гнёздово_Селище_1|EtGr|карта_05", "216_Гнёздово_Селище_2|EtGr|карта_04_1.0", "212_Гнёздово_Усадьба|TerFl|"]
    picks = [p for p in picks if os.path.exists(f"work/geo/objects_oof/{p.replace('|', '__')}.npz")]
    fig, axs = plt.subplots(1, len(picks), figsize=(6 * len(picks), 9), squeeze=False)
    for k, uid in enumerate(picks):
        u = U[uid]; d = GD.load_unit(u, GSD[u["method"]]); z = np.load(f"work/geo/objects_oof/{uid.replace('|', '__')}.npz")
        prob = z["prob"].astype(np.float32); v = z["valid"]
        m = (ndi.gaussian_filter(prob, 1.0) >= 0.4) & v; m = ndi.binary_opening(m); lab, n = ndi.label(m)
        means = ndi.mean(prob, lab, np.arange(1, n + 1)) if n else []
        preds = []
        for geom, val in RF.shapes(lab.astype(np.int32), mask=lab > 0, transform=d["T"]):
            g = shape(geom).buffer(0)
            if g.area >= 0.3:
                preds.append((g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area), float(means[int(val) - 1])))
        tp, fp, fn, pairs = soft_match(preds, d["objects"])
        mp = {id(p) for p, _ in pairs}; mg = {id(q) for _, q in pairs}
        T = d["T"]; ext = (T.c, T.c + T.a * v.shape[1], T.f + T.e * v.shape[0], T.f)
        ax = axs[0, k]
        rgb = np.moveaxis(np.clip(d["rgb"], 0, 255).astype(np.uint8), 0, -1).copy(); rgb[~d["valid"]] = 255
        ax.imshow(rgb, extent=ext)
        for q in d["objects"]:
            outline(ax, q, "white" if id(q) in mg else C_FN, 1.4, "--")
        for p, _ in preds:
            outline(ax, p, C_TP if id(p) in mp else C_FP, 1.4)
        P_, R_, F_ = prf(tp, fp, fn)
        ax.set_title(f"{u['region']} {u['method']} {u['key']}\nTP {tp}, FP {fp}, missed {fn}; F1 {F_:.2f}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Geophysics objects, leave-one-region-out pixel LightGBM (threshold 0.4): green = found object, white dashed = its label, "
                 "red = false alarm, yellow dashed = missed label", fontsize=12)
    plt.tight_layout(); plt.savefig("docs/figures/det_geo_objects.png", dpi=90); print("saved det_geo_objects.png")


if __name__ == "__main__":
    {"lidar": lidar, "finds": finds, "objects": objects}[sys.argv[1]]()
