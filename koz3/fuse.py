"""Cross-modal fusion for regions observed by several data types (LAS + SpOr, LAS + Ae, SpOr + Ae, ...).

Labels are kept per data type, but the metric matches by class only (markup_type is ignored), and 54-61 % of
LiDAR objects in the training data are also labelled at the same spot in the satellite / aerial layer.
So, per region and per pair of data types (A, B):
  * agreement  — a detection in A with a same-family detection in B within MATCH_M: both confidences are
                 raised: c' = 1 - (1 - c)(1 - beta * c_partner)
  * twin       — a detection in A with no partner in B is copied into B with confidence alpha_AB * c
                 (it can hit B's label of the same physical object).
alpha / beta are tuned out-of-fold on training sites that carry both label types.
"""
import numpy as np

MATCH_M = 6.0
OPTICAL = ("li", "sp_or", "ae", "or")
KURGAN = ("kurgany_povrezhdennye", "kurgany_tselye")


def _family(c):
    return "kurgan" if c in KURGAN else c


def _xy(f):
    from shapely.geometry import shape
    p = shape(f["geometry"]).centroid
    return p.x, p.y


def fuse(features, alpha=None, beta=0.5, match_m=MATCH_M):
    """features: list of output features (EPSG:3857). alpha: {(A, B): factor} for twin emission (missing → none)."""
    import copy
    from scipy.spatial import cKDTree
    alpha = alpha or {}
    out = list(features)
    by = {}
    for i, f in enumerate(features):
        p = f["properties"]
        if p.get("markup_type") in OPTICAL:
            by.setdefault(p["region_name"], {}).setdefault(p["markup_type"], []).append(i)
    boost = np.zeros(len(features))
    twins = []
    for region, mods in by.items():
        if len(mods) < 2:
            continue
        xy = {m: np.array([_xy(features[i]) for i in idx]) for m, idx in mods.items()}
        for A in mods:
            for B in mods:
                if A == B:
                    continue
                k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(xy[B][:, 1].mean() / 6378137.0)) - np.pi / 2)))
                tree = cKDTree(xy[B])
                for ia, (x, y) in zip(mods[A], xy[A]):
                    fa = features[ia]["properties"]
                    near = tree.query_ball_point((x, y), match_m * k)
                    partners = [mods[B][j] for j in near if _family(features[mods[B][j]]["properties"]["class_name"]) == _family(fa["class_name"])]
                    if partners:
                        cb = max(features[j]["properties"]["confidence"] for j in partners)
                        boost[ia] = max(boost[ia], cb)
                    elif (A, B) in alpha and alpha[(A, B)] > 0:
                        t = copy.deepcopy(features[ia])
                        t["properties"]["markup_type"] = B
                        t["properties"]["confidence"] = float(alpha[(A, B)] * fa["confidence"])
                        twins.append(t)
    for i, b in enumerate(boost):
        if b > 0:
            c = out[i]["properties"]["confidence"]
            out[i] = dict(out[i], properties=dict(out[i]["properties"], confidence=float(1 - (1 - c) * (1 - beta * b))))
    return out + twins
