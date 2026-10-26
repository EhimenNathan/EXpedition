"""Figures comparing geophysics finds v2 (honest F1 0.685) and v3 (honest F1 0.726), and v3 on every map by style.

    python scripts/fig_finds_compare.py

All points are out-of-fold (leave-one-region-out): the model that scored a region never saw it. Each version uses its
honest per-region threshold (chosen on the other regions), i.e. exactly the predictions that produced the reported
scores. Green circle = find detected (official rule: predicted centre within 2 m), red circle = false alarm,
yellow x = missed find. Writes docs/figures/finds_v2_vs_v3_*.png and finds_v3_by_style.png."""
import json, os, sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import geo_data as GD
from koz3.official.fastmatch import soft_match, prf

C_TP, C_FP, C_FN = "#16a34a", "#dc2626", "#eab308"
V2 = pd.read_parquet("work/geo/finds_candidates.parquet"); V2R = pd.read_csv("work/geo/finds_report.csv")
V3 = pd.read_parquet("work/geo/finds_v3/oof.parquet")
GT3 = {k: np.array(v) for k, v in json.load(open("work/geo/finds_v3/gt.json", encoding="utf-8")).items()}
U = {f"{u['region']}|{u['key']}": u for u in GD.discover("data_geo") if u["method"] == "MagGr" and u["finds"]}
THR3 = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def honest_thr_v3():
    """Per-region threshold for v3 p_base chosen on the other regions (as in geo_finds_v3 model)."""
    res = {}
    for uid, g in V3.groupby("uid"):
        gts = [Point(x, y) for x, y in GT3[uid]]
        for t in THR3:
            s = g[g.p_base >= t]
            res[(uid, t)] = soft_match([(Point(x, y).buffer(0.25, 8), p) for x, y, p in zip(s.x, s.y, s.p_base)], gts)[:3]
    uids = sorted(V3.uid.unique()); reg = {u: u.split("|")[0] for u in uids}; out = {}
    for r in sorted(set(reg.values())):
        oth = [u for u in uids if reg[u] != r]
        out[r] = max(THR3, key=lambda t: prf(*np.sum([res[(u, t)] for u in oth], axis=0))[2])
    return out


T3 = honest_thr_v3()
T2 = {r.region: r.thr for r in V2R[(V2R.model == "p") & V2R.region.str[:3].str.isdigit()].itertuples()}


def preds(version, uid):
    region = uid.split("|")[0]
    if version == "v2":
        g = V2[V2.uid == uid]
        if not len(g) or region not in T2:
            return None
        s = g[g.p >= T2[region]]; return list(zip(s.x, s.y, s.p)), T2[region]
    g = V3[V3.uid == uid]; s = g[g.p_base >= T3[region]]; return list(zip(s.x, s.y, s.p_base)), T3[region]


def draw(ax, d, P, gts, zoom=None, size=10):
    T = d["T"]; img = np.where(d["valid"], d["lum"], np.nan)
    ext = (T.c, T.c + T.a * img.shape[1], T.f + T.e * img.shape[0], T.f)
    ax.imshow(img, cmap="gray", extent=ext)
    pr = [(Point(x, y).buffer(0.25, 8), p) for x, y, p in P]
    tp, fp, fn, pairs = soft_match(pr, gts)
    hit = {id(p) for p, _ in pairs}; hg = {id(q) for _, q in pairs}
    ax.scatter([p.centroid.x for p, _ in pr if id(p) in hit], [p.centroid.y for p, _ in pr if id(p) in hit], s=size, facecolors="none", edgecolors=C_TP, linewidths=1.2)
    ax.scatter([p.centroid.x for p, _ in pr if id(p) not in hit], [p.centroid.y for p, _ in pr if id(p) not in hit], s=size, facecolors="none", edgecolors=C_FP, linewidths=1.2)
    miss = [q for q in gts if id(q) not in hg]
    ax.scatter([q.x for q in miss], [q.y for q in miss], s=size, marker="x", c=C_FN, linewidths=1.2)
    if zoom:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
    ax.set_xticks([]); ax.set_yticks([])
    return tp, fp, fn


def compare(uids, fname, title):
    fig, axs = plt.subplots(len(uids), 4, figsize=(22, 5.6 * len(uids)), squeeze=False)
    for row, uid in enumerate(uids):
        d = GD.load_unit(U[uid], 0.1); gts = [Point(x, y) for x, y in GT3[uid]]
        g3 = V3[(V3.uid == uid) & (V3.tgt == 1)]
        cx, cy = (np.median(g3.x), np.median(g3.y)) if len(g3) else (np.mean(V3[V3.uid == uid].x), np.mean(V3[V3.uid == uid].y))
        zoom = (cx - 12, cx + 12, cy - 12, cy + 12)
        for col, (ver, zm) in enumerate((("v2", None), ("v3", None), ("v2", zoom), ("v3", zoom))):
            ax = axs[row, col]; r = preds(ver, uid)
            if r is None:
                ax.text(0.5, 0.5, f"{ver}: region not scored\n(skipped by out-of-memory in v2)", ha="center", va="center", transform=ax.transAxes); ax.set_axis_off(); continue
            P, thr = r; tp, fp, fn = draw(ax, d, P, gts, zm, 22 if zm else 8)
            p_, r_, f_ = prf(tp, fp, fn)
            ax.set_title(f"{uid.split('|')[0]} {uid.split('|')[1]}\n{ver}{' zoom 24 m' if zm else ''}: F1 {f_:.3f} (P {p_:.2f} R {r_:.2f}), "
                         f"TP {tp} FP {fp} miss {fn}, thr {thr}", fontsize=10)
    fig.suptitle(title + "\ngreen = find detected (centre within 2 m), red = false alarm, yellow x = missed find; all out-of-fold", fontsize=13)
    plt.tight_layout(); plt.savefig(f"docs/figures/{fname}", dpi=80); plt.close(fig); print("saved", fname, flush=True)


def by_style():
    uids = sorted(V3.uid.unique())
    fig, axs = plt.subplots(3, 5, figsize=(26, 16)); axs = axs.ravel()
    for ax, uid in zip(axs, uids):
        d = GD.load_unit(U[uid], 0.1); gts = [Point(x, y) for x, y in GT3[uid]]
        P, thr = preds("v3", uid); tp, fp, fn = draw(ax, d, P, gts, None, 5)
        style = "CRISP" if uid.startswith(("201", "202", "203", "204", "205", "206", "207")) else ("COLOUR" if "цвет" in uid else "SMOOTH")
        ax.set_title(f"[{style}] {uid.split('|')[0][:26]} {uid.split('|')[1]}\nF1 {prf(tp, fp, fn)[2]:.3f}  TP {tp} FP {fp} miss {fn}", fontsize=10)
    for ax in axs[len(uids):]:
        ax.set_axis_off()
    fig.suptitle("Finds v3 (honest F1 0.726) on every magnetometry map, out-of-fold. CRISP = Семибратнее (dipoles visible), "
                 "SMOOTH/COLOUR = Гнёздово (interpolated / colour-rendered)", fontsize=14)
    plt.tight_layout(); plt.savefig("docs/figures/finds_v3_by_style.png", dpi=70); plt.close(fig); print("saved finds_v3_by_style.png")


def bars():
    r3 = pd.read_csv("work/geo/finds_v3/report.csv").set_index("variant").T
    regs = [c for c in r3.index if c[:3].isdigit()]
    v3 = r3.loc[regs, "base"].astype(float); v2 = pd.Series({r: V2R[(V2R.model == "p") & (V2R.region == r)].F1.iloc[0] if (V2R.region == r).any() else np.nan for r in regs})
    phys = r3.loc[regs, "phys"].astype(float)
    x = np.arange(len(regs)); fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 0.27, phys, 0.27, label="untrained physics", color="#9ca3af"); ax.bar(x, v2, 0.27, label="v2 (0.685)", color="#60a5fa"); ax.bar(x + 0.27, v3, 0.27, label="v3 (0.726)", color="#2563eb")
    ax.axvspan(-0.5, 6.5, color="#dcfce7", alpha=0.5, zorder=0); ax.axvspan(6.5, len(regs) - 0.5, color="#fee2e2", alpha=0.5, zorder=0)
    ax.text(3, 0.95, "CRISP maps (Семибратнее 201-207)", ha="center"); ax.text(9, 0.95, "SMOOTH / COLOUR maps (Гнёздово 208-212)", ha="center")
    ax.set_xticks(x); ax.set_xticklabels([r.split("_")[0] + "\n" + "_".join(r.split("_")[1:])[:14] for r in regs], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("official F1 (finds), honest"); ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.set_title("Geophysics finds per region: untrained physics vs v2 vs v3 (v2 skipped 205 and 207 by out-of-memory)")
    plt.tight_layout(); plt.savefig("docs/figures/finds_v2_vs_v3_bars.png", dpi=90); plt.close(fig); print("saved finds_v2_vs_v3_bars.png")


bars()
compare(["204_Семибратнее_Городище_4|", "201_Семибратнее_Городище_1|"], "finds_v2_vs_v3_crisp.png",
        "CRISP maps (Семибратнее): finds v2 vs v3")
compare(["209_Гнёздово_Терраса_1|", "212_Гнёздово_Усадьба|цвет_2"], "finds_v2_vs_v3_smooth_colour.png",
        "SMOOTH (209) and COLOUR (212) maps (Гнёздово): finds v2 vs v3")
by_style()
