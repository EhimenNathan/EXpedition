"""Training-fold-only aerial annotation audit for future sampling decisions."""
import collections
import json
import zlib
from pathlib import Path

import numpy as np

ROOT = Path('/home/nathan/Expedition Competition')
HERE = Path(__file__).resolve().parent
rows = json.loads((ROOT/'data/ae_prep/ann.json').read_text())
train = [r for r in rows if r['fold'] != 0 and
         (r['boxes'] or zlib.crc32(r['file'].encode()) / 2**32 < .2)]
sites = {}
for r in train:
    s = sites.setdefault(r['region'], {'chips': 0, 'class_counts': collections.Counter(), 'short_sides': []})
    s['chips'] += 1
    s['class_counts'].update(r['classes'])
    s['short_sides'].extend(min(b[2]-b[0], b[3]-b[1]) for b in r['boxes'])
out = {'train_chips': len(train), 'train_sites': len(sites),
       'boxes': sum(len(r['boxes']) for r in train),
       'class_counts': dict(collections.Counter(c for r in train for c in r['classes'])),
       'sites': {name: {'chips': s['chips'], 'boxes': len(s['short_sides']),
                        'class_counts': dict(s['class_counts']),
                        'median_short_side_px': float(np.median(s['short_sides'])) if s['short_sides'] else None}
                 for name, s in sorted(sites.items())}}
(HERE/'train_distribution_fold0.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print('training chips/sites/boxes', out['train_chips'], out['train_sites'], out['boxes'])
print('class counts', out['class_counts'])
