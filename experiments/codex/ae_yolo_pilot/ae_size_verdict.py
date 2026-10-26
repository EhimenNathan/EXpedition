"""Official-metric Ae diagnostic with class-specific damaged-kurgan size recall.

Use a threshold committed before inspecting a candidate fold. Prediction and
ground-truth coordinates are converted by the official scorer to metric UTM.
This script reads models and labels, but never trains or changes them.
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

ROOT = Path('/home/nathan/Expedition Competition')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import CALC, score
from sp_fusion_oof import fc

DAMAGED = 'kurgany_povrezhdennye'
BINS = [100., 500., 1000., 5000., 10000., float('inf')]


def size_matches(pred_fc, gt_fc):
    """Same confidence order, candidates, match quality and tie-break as scorer."""
    pred = CALC._collect_by_class(pred_fc).get(DAMAGED, [])
    gt = CALC._collect_by_class(gt_fc, preserve_points=True).get(DAMAGED, [])
    geoms = [g for g, _ in gt]
    tree = STRtree(geoms) if geoms else None
    matched = set()
    for p, conf in sorted(pred, key=lambda x: x[1], reverse=True):
        best, best_j = float('-inf'), -1
        for j in sorted(map(int, tree.query(p))) if tree is not None else []:
            if j in matched:
                continue
            hit, quality = CALC._match_quality(p, geoms[j])
            if hit and quality > best + 1e-9:
                best, best_j = quality, j
        if best_j >= 0:
            matched.add(best_j)
    total = np.zeros(len(BINS), int)
    found = np.zeros(len(BINS), int)
    for j, g in enumerate(geoms):
        if g.area <= 0:
            continue
        b = int(np.searchsorted(BINS, g.area, side='left'))
        total[b] += 1
        found[b] += j in matched
    return len(matched), total, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--threshold', type=float, default=.4)
    ap.add_argument('--folds', default='0,1,2,3')
    a = ap.parse_args()
    wanted = {int(x) for x in a.folds.split(',')}
    D = pd.read_parquet(a.pred)
    assert {'region','cls','score','x','y','fold'}.issubset(D.columns)
    D = D[D.fold.isin(wanted)]
    manifest=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    folds={r['region']:r['fold'] for r in manifest if r['fold'] in wanted}
    assert set(D.region).issubset(folds)
    L = gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L = L[(L['mod']=='Ae') & L.region.isin(folds)]
    for fold in sorted(wanted):
        sites = {r for r, f in folds.items() if f == fold}
        gt = to_gt_geojson(L[L.region.isin(sites)])
        pred = fc(D[(D.fold==fold)&(D.score>=a.threshold)])
        overall, res = score(pred, gt)
        matched, total, found = size_matches(pred, gt)
        c = res[DAMAGED]
        assert matched == c['tp'], (matched, c['tp'])
        print('fold',fold,'sites',len(sites),'AeF1',round(overall,5),
              'damaged P',round(c['p'],5),'R',round(c['r'],5),
              'TP/FP/FN',c['tp'],c['fp'],c['fn'])
        print(' damaged size bins m2 <=',BINS,'GT',total.tolist(),
              'TP',found.tolist(),'recall',
              [round(float(t/g),3) if g else None for t,g in zip(found,total)])


if __name__ == '__main__':
    main()
