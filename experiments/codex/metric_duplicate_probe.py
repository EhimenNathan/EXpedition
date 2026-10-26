#!/usr/bin/env python3
"""Minimal proof of how the official KOZ-3 matcher handles repeated GT objects."""

from copy import deepcopy

from koz3.score import score


CLASS = "kurgany_povrezhdennye"


def feature(coords, confidence=1.0, markup_type=None):
    props = {"class_name": CLASS, "confidence": confidence}
    if markup_type is not None:
        props["markup_type"] = markup_type
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


square = [[30.0, 50.0], [30.0001, 50.0], [30.0001, 50.0001], [30.0, 50.0001], [30.0, 50.0]]
gt_one = feature(square, markup_type="Li")
gt_two = feature(square, markup_type="Ae")
pred = feature(square)

for n_pred in (1, 2, 3):
    gt = {"type": "FeatureCollection", "features": [deepcopy(gt_one), deepcopy(gt_two)]}
    pr = {"type": "FeatureCollection", "features": [deepcopy(pred) for _ in range(n_pred)]}
    value, details = score(pr, gt, classes=[CLASS])
    d = details[CLASS]
    print(f"pred={n_pred} gt=2 score={value:.6f} tp={d['tp']} fp={d['fp']} fn={d['fn']}")
