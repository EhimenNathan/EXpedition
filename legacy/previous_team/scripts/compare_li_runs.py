"""Compare LiDAR OOF runs on the same sites under the official metric (koz3-1.5.0).

    python scripts/compare_li_runs.py A2_ref_unet_r34_all A3_r34_all59_8k

For each pool (sites every run predicted; sites new in KOZ-3, >= 179) and each run: the best single (mask, conf) setting,
its per-class F1, and the per-site F1 of kurgans pooled, so a gain can be checked for consistency across sites, not
only in the total. Reads kaggle/out/gpu_loss/<tag>/oof_pred.(parquet|pkl) via kaggle/official_rescore.py."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
import official_rescore as R

tags = sys.argv[1:]
runs = {t: R.load(t) for t in tags}
common = sorted(set.intersection(*[set(P.group) | set(G.group) for P, G in runs.values()]))
K = ("kurgany_tselye", "kurgany_povrezhdennye")


def best(P, G):
    b = None
    for m in sorted(P.thr.unique()):
        for c in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            f, rows = R.score(P[P.thr == m], G, c)
            if b is None or f > b[0]:
                b = (f, m, c, {k: round(v["F1"], 3) for k, v in rows.items() if v["n_gt"]})
    return b


for name, pool in (("common", common), ("new in KOZ-3 (>=179)", [s for s in common if int(s[:3]) >= 179])):
    print(f"\n## {name}: {len(pool)} sites")
    cfg = {}
    for t, (P, G) in runs.items():
        f, m, c, rows = best(P[P.group.isin(pool)], G[G.group.isin(pool)]); cfg[t] = (m, c)
        print(f"  {t:28s} final {f:.4f} (mask {m}, conf>={c})  {rows}")
    if name == "common" and len(tags) == 2:  # per-site kurgan F1 at each run's own best setting
        a, b = tags; wins = 0; n = 0; deltas = []
        for s in pool:
            fs = []
            for t in (a, b):
                P, G = runs[t]; m, c = cfg[t]
                Ps, Gs = P[(P.group == s) & (P.thr == m)], G[G.group == s]
                if not Gs.lcls.isin(K).any():
                    break
                _, rows = R.score(Ps.assign(lcls=Ps.lcls.where(~Ps.lcls.isin(K), K[0])), Gs.assign(lcls=Gs.lcls.where(~Gs.lcls.isin(K), K[0])), c)
                fs.append(rows.get(K[0], {}).get("F1", 0.0))
            if len(fs) == 2:
                n += 1; wins += fs[1] > fs[0] + 0.01; deltas.append(fs[1] - fs[0])
        print(f"  per-site kurgan-any F1, {b} vs {a}: better on {wins}/{n} sites (>0.01), median delta {np.median(deltas):+.3f}")
