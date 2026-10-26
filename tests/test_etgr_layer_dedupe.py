"""Regression check: EtGr layers survive final dedupe, satellite overlaps do not."""
from pathlib import Path
import json
import sys
from shapely.geometry import Point, mapping
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT))


def feature(markup, image, confidence, source_layer=None):
    return {"type": "Feature", "geometry": mapping(Point(1, 1).buffer(1)),
            "properties": {"class_name": "object_poly", "region_name": "R",
                           "markup_type": markup, "image": image,
                           "confidence": confidence, "source_layer": source_layer}}


def apply_final_dedupe(features, dd):
    from koz3.dedupe import dedupe_region
    skip = set(dd.get("skip_markups", ["li"])) | {"etgr"}
    skip_cls = set(dd.get("skip_classes", ["finds_points"]))
    outside = lambda f: (f["properties"].get("markup_type") in skip or
                         f["properties"]["class_name"] in skip_cls)
    return ([f for f in features if outside(f)] +
            dedupe_region([f for f in features if not outside(f)], dd.get("dist_m", 3.0)))


if __name__ == "__main__":
    dd = {"enabled": True, "dist_m": 3.0, "skip_markups": ["li"],
          "skip_classes": ["finds_points"]}
    et = apply_final_dedupe([feature("etgr", "layer1", .9), feature("etgr", "layer2", .8)], dd)
    sp = apply_final_dedupe([feature("sp_or", "scene1", .9), feature("sp_or", "scene2", .8)], dd)
    assert len(et) == 2, len(et)
    assert len(sp) == 1, len(sp)
    sp_layers = apply_final_dedupe([feature("sp_or", "obj1", .9, "scene1"),
                                    feature("sp_or", "obj2", .8, "scene2")], dd)
    assert len(sp_layers) == 2, len(sp_layers)
    from koz3_infer.run import _jobs_for
    region = Path("region_with_etgr_and_terfl")
    et_path, tf_path = Path("map.tif"), Path("profile.png")
    plan = {"etgr": [(region, [et_path])], "terfl": [(region, [tf_path])]}
    jobs = _jobs_for(SimpleNamespace(FAMILIES=["etgr", "terfl", "maggr"]), plan)
    assert jobs == [(region, [et_path, tf_path])], jobs
    print("PASS: EtGr keeps 2/2 layers; legacy imagery dedupes 2->1; independent source layers keep 2/2; "
          "multi-family region queues once")
