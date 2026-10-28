"""Nested OOF evaluation of the ring-aware LiDAR model as a kurgan specialist.

The deployed blend has seven channels: three heat maps (kurgan-any, hillfort,
fortification) followed by four class masks.  A ring-specific augmentation is
intended to improve only kurgans.  Averaging every channel can therefore dilute
the stronger deployed hillfort/fortification predictions.  This script tests
three prospective fusion scopes while keeping decoding identical to deployment:

  heat       mix only kurgan-any heat
  heat_masks mix kurgan-any heat and damaged/intact masks
  all        conventional whole-tensor blend (cross-check)

Fusion scope, candidate weight and threshold are selected on the other
geographic folds.  The zero-weight deployed setting is included in the grid.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SCOPES = {
    "heat": (0,),
    "heat_masks": (0, 3, 4),
    "all": tuple(range(7)),
}
WEIGHTS = (0.0, 0.5, 1.0, 2.0)
THRESHOLDS = (0.12, 0.15, 0.18)
GATE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def main(candidate="work/li/oof_augxr.pkl"):
    from koz3.labels import to_gt_geojson
    from koz3.li_train import CLS, decode_site_v2, labels, site_items
    from koz3.nested import _add_all, fold_counts, nested, weighted
    from robust_accept import report, site_counts
    import li_compare_oof as lc

    lab = labels()
    items = {it["region"]: it for it in site_items(lab)}
    with open(os.path.join(ROOT, "work/li/oof_blend3.pkl"), "rb") as f:
        base = pickle.load(f)
    ring = lc._pk(candidate)
    sites = sorted(set(base) & set(ring))
    fold = {r: items[r]["fold"] for r in sites}
    folds = sorted(set(fold.values()))
    prior = {"R": 200.0, "K": 3, "snb": 0.2}
    cache = {}

    # Label-free site statistic: mean of the strongest 0.1% positive changes in
    # kurgan heat.  A ring specialist should diverge most on ring-like terrain.
    # Convert it to a global rank so the gate is invariant to score calibration.
    divergence = {}
    for region in sites:
        delta = ring[region][0].astype(np.float32) - base[region][0].astype(np.float32)
        valid_delta = delta[items[region]["valid"].astype(bool)]
        k = max(1, int(np.ceil(valid_delta.size * 0.001)))
        divergence[region] = float(np.partition(valid_delta, -k)[-k:].mean())
    ordered = sorted(sites, key=lambda r: divergence[r])
    rank = {r: (i + 1) / len(ordered) for i, r in enumerate(ordered)}
    print("ring divergence quantiles", np.quantile(list(divergence.values()), [0, .25, .5, .75, 1]).round(5), flush=True)

    def probability(setting, region):
        scope, weight, _, gate_fraction = setting
        out = base[region].astype(np.float32).copy()
        active = rank[region] > 1.0 - gate_fraction
        if weight and active:
            idx = SCOPES[scope]
            out[list(idx)] = (
                3.0 * out[list(idx)] + weight * ring[region].astype(np.float32)[list(idx)]
            ) / (3.0 + weight)
        return out

    def decode(setting, region):
        key = (setting, region)
        if key not in cache:
            _, _, threshold, _ = setting
            it = items[region]
            cache[key] = decode_site_v2(
                probability(setting, region),
                it["valid"],
                dict(it["meta"], region=region),
                thr=threshold,
                dual=0.05,
                comp_thr=0.5,
                prior=prior,
            )
        return cache[key]

    deployed = ("heat", 0.0, 0.15, 1.0)
    settings = [deployed] + [
        (scope, w, t, gate_fraction)
        for scope in SCOPES
        for w in WEIGHTS if w > 0
        for t in THRESHOLDS
        for gate_fraction in GATE_FRACTIONS
    ]
    gt = {
        f: to_gt_geojson(lab[lab.region.isin([r for r in sites if fold[r] == f])])
        for f in folds
    }
    counts = fold_counts(
        lambda setting, f: {
            "type": "FeatureCollection",
            "features": [g for r in sites if fold[r] == f for g in decode(setting, r)],
        },
        settings,
        folds,
        gt,
        CLS,
    )
    for setting in sorted(settings, key=lambda s: -weighted(_add_all(counts[s].values())))[:12]:
        print(f"{setting}: pooled {weighted(_add_all(counts[setting].values())):.5f}", flush=True)
    honest, chosen, oracle, oracle_setting, _ = nested(counts)
    deployed_score = weighted(_add_all(counts[deployed].values()))
    print(
        f"deployed {deployed_score:.5f} | nested honest {honest:.5f} "
        f"(oracle {oracle:.5f} @ {oracle_setting}) chosen {list(chosen.values())}",
        flush=True,
    )
    changes = {
        f: round(weighted(counts[chosen[f]][f]) - weighted(counts[deployed][f]), 5)
        for f in folds
    }
    print("per-fold change vs deployed:", changes, flush=True)

    gt_site = {r: to_gt_geojson(lab[lab.region == r]) for r in sites}
    site = site_counts(
        lambda setting, r: {"type": "FeatureCollection", "features": decode(setting, r)},
        (deployed, oracle_setting),
        sites,
        gt_site,
        CLS,
    )
    report(f"ring-aware channel stack {deployed} -> {oracle_setting}", site[deployed], site[oracle_setting])


if __name__ == "__main__":
    main(*sys.argv[1:])
