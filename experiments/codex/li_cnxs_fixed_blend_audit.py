"""Fixed, read-only OOF audit of a 25% ConvNeXt-S LiDAR addition.

No threshold or blend search: compare v13's 3-map blend with a predeclared
75% v13 / 25% ConvNeXt-S map blend at the exact v13 decoder settings.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/nathan/Expedition Competition')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import li_compare_oof as lc
from koz3.labels import to_gt_geojson
from koz3.li_train import decode_site_v2, labels, site_items
from koz3.score import score, WEIGHTS

PRIOR = {'R': 200, 'K': 3, 'snb': .2}


def decode(arr, it, region):
    return decode_site_v2(arr.astype(np.float32), it['valid'],
                          dict(it['meta'], region=region),
                          thr=.15, dual=.05, comp_thr=.5, prior=PRIOR)


def main():
    base = lc.load('ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst')
    added = lc.load('work/li/oof_cnxs.pkl')
    assert set(base) == set(added), (len(base), len(added))
    lab = labels()
    items = {it['region']: it for it in site_items(lab)}
    feats = {'base': {}, 'plus': {}}
    for j, (region, p) in enumerate(base.items(), 1):
        it = items[region]
        q = (.75 * p.astype(np.float32) + .25 * added[region].astype(np.float32))
        feats['base'][region] = decode(p, it, region)
        feats['plus'][region] = decode(q, it, region)
        print('decoded', j, len(base), region, len(feats['base'][region]),
              len(feats['plus'][region]), flush=True)
    all_regions = list(base)
    fold = {r: int(items[r]['fold']) for r in all_regions}
    out = {'rule': '75% v13 maps + 25% ConvNeXt-S, fixed prior R200/K3/snb.2 and thr.15',
           'folds': {}, 'pooled': {}}
    for key in sorted(set(fold.values())) + ['all']:
        regs = all_regions if key == 'all' else [r for r in all_regions if fold[r] == key]
        gt = to_gt_geojson(lab[lab.region.isin(regs)])
        dest = out['pooled'] if key == 'all' else out['folds'].setdefault(str(key), {})
        for name in ('base', 'plus'):
            fc = {'type': 'FeatureCollection', 'features': [f for r in regs for f in feats[name][r]]}
            val, classes = score(fc, gt)
            dest[name] = {'f1': float(val), 'objects': len(fc),
                          'classes': {c: {k: float(v[k]) if k == 'f1' else int(v[k])
                                          for k in ('f1', 'tp', 'fp', 'fn')}
                                      for c, v in classes.items()}}
        dest['delta'] = dest['plus']['f1'] - dest['base']['f1']
        print('fold', key, 'base', dest['base']['f1'], 'plus', dest['plus']['f1'],
              'delta', dest['delta'], flush=True)
    path = Path(__file__).with_name('li_cnxs_fixed_blend_audit.json')
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print('saved', path, flush=True)


if __name__ == '__main__':
    main()
