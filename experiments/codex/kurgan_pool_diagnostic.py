"""Read-only LiDAR OOF diagnostic: localization versus intact/damaged classification.

Uses the fixed v12-style ensemble and deployed decoder with no parameter search.
Relabels both kurgan classes to one only for the diagnostic score; this is not
a submission candidate or a substitute for the official two-class metric.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/nathan/Expedition Competition')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from koz3.labels import to_gt_geojson
from koz3.li_train import decode_site_v2, labels, site_items
from koz3.score import score
import li_compare_oof as lc

DAMAGED = 'kurgany_povrezhdennye'
INTACT = 'kurgany_tselye'
KURGAN = {DAMAGED, INTACT}


def relabel(fc):
    return {
        'type': 'FeatureCollection',
        'features': [
            {'type': 'Feature', 'geometry': f['geometry'],
             'properties': dict(f['properties'], class_name=DAMAGED)}
            for f in fc['features']
            if f['properties']['class_name'] in KURGAN
        ],
    }


def main():
    arrays = lc.load('ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst')
    lab = labels()
    items = {it['region']: it for it in site_items(lab)}
    rows = []
    for region, arr in arrays.items():
        it = items[region]
        pred = {'type': 'FeatureCollection', 'features': decode_site_v2(
            arr.astype(np.float32), it['valid'], dict(it['meta'], region=region),
            thr=0.15, dual=0.05, comp_thr=0.5)}
        gt = to_gt_geojson(lab[(lab.region == region) & lab.class_name.isin(KURGAN)])
        _, separate = score(pred, gt, classes=[DAMAGED, INTACT])
        _, pooled = score(relabel(pred), relabel(gt), classes=[DAMAGED])
        rows.append({
            'region': region, 'fold': int(it['fold']),
            'damaged': {k: separate[DAMAGED][k] for k in ('tp', 'fp', 'fn', 'f1')},
            'intact': {k: separate[INTACT][k] for k in ('tp', 'fp', 'fn', 'f1')},
            'pooled': {k: pooled[DAMAGED][k] for k in ('tp', 'fp', 'fn', 'f1')},
        })
    summary = {}
    for fold in sorted({r['fold'] for r in rows}):
        sub = [r for r in rows if r['fold'] == fold]
        counts = {}
        for key in ('damaged', 'intact', 'pooled'):
            tp = sum(r[key]['tp'] for r in sub)
            fp = sum(r[key]['fp'] for r in sub)
            fn = sum(r[key]['fn'] for r in sub)
            counts[key] = {'tp': tp, 'fp': fp, 'fn': fn,
                           'f1': 2 * tp / (2 * tp + 0.5 * fp + fn) if tp else 0.0}
        summary[str(fold)] = counts
    out = Path(__file__).with_name('kurgan_pool_diagnostic.json')
    out.write_text(json.dumps({'configuration': 'v12-style LiDAR ensemble, exact decoder thr=.15 dual=.05 comp=.5',
                               'folds': summary, 'sites': rows}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('saved', out)


if __name__ == '__main__':
    main()
