"""Cross-fitted LiDAR hillfort presence stack for DINO relief localization.

The frozen DINO localizer supplies an independent location but its raw scene
scores are only moderately calibrated.  This test combines those scores with
the deployed LiDAR model's hillfort heat, mask and decoded confidence.  The
presence classifier is trained on other geographic folds only.  Existing
emissions are preserved; the conservative mode adds at most one hillfort to a
site where the deployed decoder emitted none.
"""
from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def circle(x, y):
    lat_scale = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
    return Point(float(x), float(y)).buffer(float(lat_scale), 12).__geo_interface__


def main(points="work/hf_dinov3/li_loc_points_oof.csv"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from koz3.labels import to_gt_geojson
    from koz3.li_train import CLS, decode_site_v2, labels, site_items
    from koz3.nested import _add_all, fold_counts, nested, weighted
    from robust_accept import report, site_counts

    with open(ROOT.parent.parent / "work/li/oof_blend3.pkl", "rb") as fh:
        probabilities = pickle.load(fh)
    lab = labels()
    items = {q["region"]: q for q in site_items(lab) if q["region"] in probabilities}
    dino = pd.read_csv(ROOT / points).set_index("region")
    sites = sorted(set(items) & set(dino.index))
    folds = sorted({items[r]["fold"] for r in sites})
    prior = {"R": 200.0, "K": 3, "snb": 0.2}
    base, rows = {}, []

    for i, region in enumerate(sites, 1):
        item = items[region]
        prob = probabilities[region].astype(np.float32)
        feats = decode_site_v2(
            prob,
            item["valid"],
            dict(item["meta"], region=region),
            thr=.15,
            dual=.05,
            comp_thr=.5,
            prior=prior,
        )
        base[region] = feats
        hillfort = [f for f in feats if f["properties"]["class_name"] == "gorodishcha"]
        valid = item["valid"].astype(bool)
        row = dino.loc[region]
        rows.append({
            "region": region,
            "fold": item["fold"],
            "y": int(bool(row.has_hillfort)),
            "map_max": float(row.map_max),
            "dino_scene": float(row.p_scene),
            "old_heat": float(prob[1][valid].max()),
            "old_mask": float(prob[5][valid].max()),
            "old_present": int(bool(hillfort)),
            "old_conf": max([float(f["properties"].get("confidence", 0)) for f in hillfort] or [0.0]),
        })
        if i % 10 == 0:
            print("decoded", i, "/", len(sites), flush=True)
    del probabilities

    frame = pd.DataFrame(rows).set_index("region")
    columns = ["map_max", "dino_scene", "old_heat", "old_mask", "old_present", "old_conf"]
    frame["p_stack"] = 0.0
    for fold in folds:
        train = frame[frame.fold != fold]
        test = frame[frame.fold == fold]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=0),
        ).fit(train[columns], train.y)
        frame.loc[test.index, "p_stack"] = model.predict_proba(test[columns])[:, 1]
    from sklearn.metrics import roc_auc_score
    for name in ("map_max", "dino_scene", "old_heat", "old_mask", "p_stack"):
        print(name, "OOF presence AUC", round(roc_auc_score(frame.y, frame[name]), 4), flush=True)

    def change(region, setting):
        mode, gate, map_gate = setting
        feats = copy.deepcopy(base[region])
        if mode == "base":
            return feats
        row = dino.loc[region]
        presence = float(frame.loc[region, "p_stack"])
        if presence < gate or float(row.map_max) < map_gate:
            return feats
        indices = [i for i, f in enumerate(feats) if f["properties"]["class_name"] == "gorodishcha"]
        if not indices and mode in ("rescue", "both"):
            feats.append({
                "type": "Feature",
                "geometry": circle(float(row.x3857), float(row.y3857)),
                "properties": {
                    "class_name": "gorodishcha",
                    "confidence": presence,
                    "region_name": region,
                    "markup_type": "li",
                },
            })
        elif indices and mode in ("replace", "both"):
            best = max(indices, key=lambda i: feats[i]["properties"].get("confidence", 0))
            feats[best]["geometry"] = circle(float(row.x3857), float(row.y3857))
        return feats

    settings = [("base", 0.0, 0.0)] + [
        (mode, gate, map_gate)
        for mode in ("rescue", "replace", "both")
        for gate in (.5, .6, .7, .8, .9)
        for map_gate in (.4, .6, .8)
    ]
    gt = {
        fold: to_gt_geojson(lab[lab.region.isin([r for r in sites if items[r]["fold"] == fold])])
        for fold in folds
    }
    counts = fold_counts(
        lambda setting, fold: {
            "type": "FeatureCollection",
            "features": sum((change(r, setting) for r in sites if items[r]["fold"] == fold), []),
        },
        settings,
        folds,
        gt,
        CLS,
    )
    base_setting = ("base", 0.0, 0.0)
    base_score = weighted(_add_all(counts[base_setting].values()))
    for setting in sorted(settings, key=lambda s: -weighted(_add_all(counts[s].values())))[:12]:
        total = _add_all(counts[setting].values())
        print(setting, "pooled", round(weighted(total), 5), "gor", total.get("gorodishcha"), flush=True)
    honest, chosen, oracle, oracle_setting, _ = nested(counts)
    print(
        "base", round(base_score, 5), "nested", round(honest, 5), "delta", round(honest - base_score, 5),
        "oracle", round(oracle, 5), oracle_setting, "chosen", chosen,
        flush=True,
    )
    print(
        "per-fold change:",
        {f: round(weighted(counts[chosen[f]][f]) - weighted(counts[base_setting][f]), 5) for f in folds},
        flush=True,
    )
    gt_site = {r: to_gt_geojson(lab[lab.region == r]) for r in sites}
    site = site_counts(
        lambda setting, r: {"type": "FeatureCollection", "features": change(r, setting)},
        (base_setting, oracle_setting),
        sites,
        gt_site,
    )
    report(f"LiDAR DINO presence stack {oracle_setting}", site[base_setting], site[oracle_setting])


if __name__ == "__main__":
    main(*sys.argv[1:])
