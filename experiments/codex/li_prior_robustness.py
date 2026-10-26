"""Fixed-threshold, per-site robustness audit of v14 LiDAR neighbour prior.

The v12 and v14 decoders differ only by the v14 neighbour prior here.  This
uses geographic OOF probability maps and the exact shipped thresholds, then
tests whether the aggregate gain depends on a few sites.  No model fitting.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

from koz3.labels import to_gt_geojson
from koz3.li_train import CLS, decode_site_v2, labels, site_items
from koz3.nested import _add_all, weighted
from koz3.score import score
from li_compare_oof import load


def counts_for(pred, gt):
    _, res = score({"type": "FeatureCollection", "features": pred}, gt)
    return {name: (int(res[name]["tp"]), int(res[name]["fp"]), int(res[name]["fn"]))
            for name in CLS if name in res}


def main():
    pred = load("ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst")
    labs = labels()
    items = {it["region"]: it for it in site_items(labs)}
    configs = {"base": None, "prior": {"R": 200, "K": 3, "snb": 0.2}}
    by_site = {}
    for n, (region, prob) in enumerate(pred.items(), 1):
        it = items[region]
        gt = to_gt_geojson(labs[labs.region == region])
        row = {"fold": int(it["fold"]), "site": region}
        for key, prior in configs.items():
            feats = decode_site_v2(prob.astype(np.float32), it["valid"],
                                   dict(it["meta"], region=region), thr=0.15,
                                   dual=0.05, comp_thr=0.5, fort_min_px=30,
                                   gor_min_px=600, r_out=1.0, prior=prior)
            row[key] = counts_for(feats, gt)
        by_site[region] = row
        if n % 10 == 0:
            print("decoded", n, "of", len(pred), flush=True)

    def score_sites(site_names, key):
        return weighted(_add_all([by_site[s][key] for s in site_names]))

    sites = sorted(by_site)
    baseline = score_sites(sites, "base")
    revised = score_sites(sites, "prior")
    individual = [(s, score_sites([s], "prior") - score_sites([s], "base")) for s in sites]
    # F1 is not additive. Rank a site's contribution by the change in the
    # *pooled* gain when that site is removed, rather than its standalone F1.
    pooled_delta = revised - baseline
    contribution = {s: pooled_delta - (
        score_sites([r for r in sites if r != s], "prior") -
        score_sites([r for r in sites if r != s], "base")) for s in sites}
    order = sorted(sites, key=lambda s: -abs(contribution[s]))
    curve = []
    for k in (0, 1, 2, 3, 5, 8, 10, 15):
        remaining = [s for s in sites if s not in order[:k]]
        curve.append({"removed": k, "n_sites": len(remaining),
                      "base": score_sites(remaining, "base"),
                      "prior": score_sites(remaining, "prior")})
    folds = sorted({row["fold"] for row in by_site.values()})
    out = {"base": baseline, "prior": revised, "delta": revised - baseline,
           "folds": {str(f): {key: score_sites([s for s in sites if by_site[s]["fold"] == f], key)
                              for key in configs} for f in folds},
           "site_deltas": sorted(individual, key=lambda x: x[1]),
           "leave_one_out_contributions": contribution,
           "leave_largest_absolute_site_out": curve,
           "site_counts": by_site}
    path = os.path.join(os.path.dirname(__file__), "li_prior_robustness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("base", "prior", "delta", "folds", "leave_largest_absolute_site_out")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
