"""Audit same-survey annotation multiplicity under the official polygon matcher.

If several GT polygons can all be matched by essentially the same footprint, ordinary
detector NMS emits one prediction and creates an avoidable recall ceiling.  This script
measures that structure without using model predictions or tuning thresholds.
"""
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

ROOT = "/home/nathan/Expedition Competition"
L = gpd.read_parquet(f"{ROOT}/data/labels_3857.parquet")


def compatible(a, b):
    if isinstance(a, Point) or isinstance(b, Point):
        return isinstance(a, Point) and isinstance(b, Point) and a.distance(b) <= 2.0
    if not isinstance(a, Polygon) or not isinstance(b, Polygon):
        return False
    try:
        inter = a.intersection(b).area
        cov = inter / max(1e-12, min(a.area, b.area))
        # Official matching also requires the prediction's representative point in GT.
        return cov >= 0.5 and (b.covers(a.representative_point()) or a.covers(b.representative_point()))
    except Exception:
        return False


def components(geoms):
    n = len(geoms)
    par = list(range(n))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: par[b] = a
    tree = STRtree(geoms)
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            j = int(j)
            if j > i and compatible(g, geoms[j]): union(i, j)
    out = defaultdict(list)
    for i in range(n): out[find(i)].append(i)
    return list(out.values())


rows = []
for (mod, region, cls), d in L.groupby(["mod", "region", "class_name"], sort=False):
    geoms = list(d.geometry)
    cs = components(geoms)
    sizes = [len(c) for c in cs]
    rows.append(dict(mod=mod, region=region, cls=cls, n=len(geoms), objects=len(cs),
                     extras=sum(s-1 for s in sizes), clusters_gt1=sum(s>1 for s in sizes),
                     max_mult=max(sizes, default=0), multiplicities=Counter(sizes)))

D = pd.DataFrame(rows)
print("SAME-MODALITY MATCH-COMPATIBLE MULTIPLICITY")
print(D.groupby("mod")[["n","objects","extras","clusters_gt1"]].sum().sort_values("extras", ascending=False).to_string())
print("\nBY MODALITY / CLASS")
print(D.groupby(["mod","cls"])[["n","objects","extras","clusters_gt1"]].sum().query("extras>0").sort_values("extras", ascending=False).to_string())
print("\nTOP REGION/CLASS GROUPS")
print(D.sort_values("extras", ascending=False).head(40)[["mod","region","cls","n","objects","extras","clusters_gt1","max_mult","multiplicities"]].to_string(index=False))

# Exact duplicate geometry is a stronger, matcher-independent subset.
E = L.assign(_wkb=L.geometry.apply(lambda g: g.wkb_hex)).groupby(["mod","region","class_name","_wkb"]).size()
print("\nEXACT DUPLICATE EXTRA LABELS", int((E-1).clip(lower=0).sum()), "groups", int((E>1).sum()), "max", int(E.max()))

D.drop(columns="multiplicities").to_parquet(f"{ROOT}/Codex Work/annotation_multiplicity_summary.parquet", index=False)
