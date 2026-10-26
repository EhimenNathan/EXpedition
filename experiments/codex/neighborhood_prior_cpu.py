"""Fixed, site-disjoint OOF audit of the proposed detection-neighborhood prior.

Reads Claude's *old* LiDAR L2 candidate table without modifying it. This is a
screening test, not v12 deployment parity. Only the preset radii/K values from
the user-provided plan are used; no threshold search or model training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from pyproj import Transformer

ROOT = Path('/home/nathan/Expedition Competition')
OUT = ROOT / 'Codex Work/neighborhood_prior_cpu.json'
sys.path.insert(0, str(ROOT))
from koz3.labels import to_gt_geojson
from koz3.score import score

SETTINGS = {
    'kurgany_povrezhdennye': (200.0, 3),
    'kurgany_tselye': (200.0, 3),
    'gorodishcha': (500.0, 2),
    'fortifikatsii': (300.0, 2),
}


def feature(row):
    # EPSG:3857 geometry; scale a true-metre radius by local Web Mercator factor.
    latitude = Transformer.from_crs(3857, 4326, always_xy=True).transform(row.x, row.y)[1]
    radius = 1.0 / np.cos(np.deg2rad(latitude))
    return {'type': 'Feature',
            'geometry': Point(row.x, row.y).buffer(radius, 12).__geo_interface__,
            'properties': {'class_name': row.cls, 'confidence': float(row.score),
                           'region_name': row.region, 'markup_type': 'li'}}


def main():
    cands = pd.read_parquet(ROOT / 'work/li/oof_prevL2_cands.parquet')
    # .3 is the widest candidate pool saved in the original table; other rows
    # repeat coordinates from higher decoder cutoffs.
    cands = cands[cands.thr == .3].copy().reset_index(drop=True)
    assert cands.region.nunique() == 45 and not cands.duplicated(['region', 'cls', 'x', 'y']).any()
    cands['adjusted'] = cands.score.to_numpy().copy()
    to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
    for (site, cls), g in cands.groupby(['region', 'cls']):
        radius_m, K = SETTINGS[cls]
        latitude = to_ll.transform(float(g.x.median()), float(g.y.median()))[1]
        radius_3857 = radius_m / np.cos(np.deg2rad(latitude))
        pts = g[['x', 'y']].to_numpy(dtype=np.float64)
        neighbors = np.array([len(z) - 1 for z in cKDTree(pts).query_ball_point(pts, radius_3857)])
        weight = .5 + .5 * np.clip(neighbors / K, 0, 2)
        cands.loc[g.index, 'adjusted'] = g.score.to_numpy() * weight
    labels = gpd.read_parquet(ROOT / 'data/labels_3857.parquet')
    labels = labels[(labels['mod'] == 'Li') & labels.region.isin(cands.region.unique())]
    results = {}
    for threshold in (.3, .4, .5):
        results[str(threshold)] = {}
        for variant in ('baseline', 'neighbor_prior'):
            keep = cands[cands[('score' if variant == 'baseline' else 'adjusted')] >= threshold]
            output = {'type': 'FeatureCollection', 'features': [feature(r) for r in keep.itertuples()]}
            gt = to_gt_geojson(labels)
            val, detail = score(output, gt)
            fold_result = {}
            for fold in sorted(cands.fold.unique()):
                sites = set(cands.loc[cands.fold == fold, 'region'])
                fs = [f for f in output['features'] if f['properties']['region_name'] in sites]
                fold_gt = to_gt_geojson(labels[labels.region.isin(sites)])
                fval, _ = score({'type': 'FeatureCollection', 'features': fs}, fold_gt)
                fold_result[str(fold)] = float(fval)
            results[str(threshold)][variant] = {
                'score': float(val), 'objects': len(output['features']),
                'fold_scores': fold_result,
                'class_counts': {k: {m: int(v[m]) for m in ('tp', 'fp', 'fn')} for k, v in detail.items()},
            }
        print('threshold', threshold, {k: round(v['score'], 6) for k, v in results[str(threshold)].items()}, flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
