"""Gate for LiDAR density augmentation: does deployed-blend OOF quality fall with point density?

docs/lidar_density.csv: ground points per occupied 1 m² cell of every training cloud (local density, 1–76 /m²),
measured from the organisers' bucket. This script decodes the deployed LiDAR blend OOF exactly as deployed
(thr 0.15, dual 0.05, comp 0.5, neighbourhood prior), scores every site with the official metric and relates kurgan
F1 / recall to density.

  python scripts/li_density_gate.py [OOF_SPEC]      # default: the deployed ens + aug + augx blend
Thinning one dense site to 0.5–4 pts/m² left mound relief almost unchanged (docs/PHYSICS_GATES_2026-10-27.md), so
the terrain model itself is not density-limited. Only if sparse sites (< 4 pts/m²) still pool clearly below dense
ones here (permutation p < 0.05) is a model-side density effect worth a thinning-augmentation run.
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")
DEFAULT = "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst"


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fn + 0.5 * fp, 1)


def main(spec=DEFAULT):
    from scipy.stats import spearmanr
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    import li_compare_oof as lc
    dens = {r["region"]: float(r["local_pts_per_m2"]) for r in csv.DictReader(open(os.path.join(ROOT, "docs/lidar_density.csv"), encoding="utf-8"))}
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    P = lc.load(spec)
    rows = []
    for r in sorted(P):
        if r not in dens:
            print("no density for", r); continue
        it = items[r]
        fs = decode_site_v2(P[r].astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05,
                            comp_thr=0.5, prior={"R": 200.0, "K": 3, "snb": 0.2})
        _, res = score({"type": "FeatureCollection", "features": fs}, to_gt_geojson(L[L.region == r]), CLS)
        tp, fp, fn = (sum(res[c][k] for c in K if c in res) for k in ("tp", "fp", "fn"))
        rows.append((r, dens[r], tp, fp, fn))
    print(f"{'site':32s} {'pts/m2':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}  kurganF1 recall")
    for r, d, tp, fp, fn in sorted(rows, key=lambda x: x[1]):
        print(f"{r[:32]:32s} {d:7.1f} {tp:5d} {fp:5d} {fn:5d}  {f1(tp, fp, fn):.3f}  {tp / max(tp + fn, 1):.3f}")
    R = [x for x in rows if x[2] + x[4] >= 10]          # sites with ≥ 10 kurgan labels
    d = np.log([x[1] for x in R]); q = np.array([f1(*x[2:]) for x in R])
    rho, _ = spearmanr(d, q)
    rng = np.random.default_rng(0)
    perm = np.array([spearmanr(d, rng.permutation(q))[0] for _ in range(5000)])
    p = (np.sum(perm >= rho) + 1) / 5001
    print(f"\n{len(R)} sites with ≥10 kurgan labels: Spearman(log density, kurgan F1) = {rho:+.3f}, one-sided permutation p = {p:.4f}")
    for lo, hi in ((0, 4), (4, 15), (15, 1e9)):
        g = [x for x in rows if lo <= x[1] < hi]
        tp, fp, fn = (sum(x[i] for x in g) for i in (2, 3, 4))
        print(f"density [{lo:>4}, {hi if hi < 1e9 else 'inf':>4}) pts/m²: {len(g):2d} sites, pooled kurgan F1 {f1(tp, fp, fn):.3f}, "
              f"recall {tp / max(tp + fn, 1):.3f} (TP {tp} FP {fp} FN {fn})")


if __name__ == "__main__":
    main(*sys.argv[1:2])
