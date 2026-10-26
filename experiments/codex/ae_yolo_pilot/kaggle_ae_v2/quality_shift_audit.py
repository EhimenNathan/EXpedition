"""Unlabelled image-quality audit of fixed fold-0 train versus held-out sites.

Held-out boxes/classes are never read. Their images are used only for a
precommitted descriptive sensor-shift diagnostic, not model selection.
"""
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/nathan/Expedition Competition')
SOURCE = ROOT / 'data/ae_prep'
OUT = ROOT / 'Codex Work/ae_yolo_pilot/kaggle_ae_v2/quality_shift.json'


def main():
    rows = json.loads((SOURCE / 'ann.json').read_text())
    groups = {'train': [], 'heldout': []}
    for r in rows:
        # Exact train-chip selection used in the Kaggle payload. Training-side
        # box presence is read only to recreate the predetermined sample.
        if r['fold'] == 0:
            side = 'heldout'
        elif r['boxes'] or zlib.crc32(r['file'].encode()) / 2**32 < .2:
            side = 'train'
        else:
            continue
        im = cv2.imread(str(SOURCE / 'chips' / r['file']), cv2.IMREAD_GRAYSCALE)
        assert im is not None
        x = im[im > 0]
        if len(x) < 100:
            continue
        p2, p50, p98 = np.percentile(x, (2, 50, 98))
        T = r['transform']
        groups[side].append({'site': r['region'], 'spread': float(p98-p2),
                             'median': float(p50), 'valid_frac': float(r['valid_frac']),
                             'gsd_x': abs(float(T[0])), 'gsd_y': abs(float(T[4]))})
    result = {}
    for side, values in groups.items():
        result[side] = {'chips': len(values), 'sites': len({v['site'] for v in values})}
        for k in ('spread', 'median', 'valid_frac', 'gsd_x', 'gsd_y'):
            a = np.array([v[k] for v in values])
            result[side][k] = {p: float(np.percentile(a,int(p))) for p in ('10','50','90')}
        result[side]['low_spread_fraction'] = float(np.mean([v['spread'] < 80 for v in values]))
        result[side]['high_spread_fraction'] = float(np.mean([v['spread'] >= 160 for v in values]))
        result[side]['per_site_spread_median'] = {
            site: float(np.median([v['spread'] for v in values if v['site']==site]))
            for site in sorted({v['site'] for v in values})}
    assert result['train']['chips']==510 and result['heldout']['chips']==298
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_site_spread_median'}
                      for k,v in result.items()},indent=2))


if __name__=='__main__':
    main()
