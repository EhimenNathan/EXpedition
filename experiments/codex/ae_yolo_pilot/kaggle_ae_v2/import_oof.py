"""Convert private Kaggle pixel detections to local geographic OOF parquet.

Held-out labels remain local and are touched only by evaluate_fold0.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine

ROOT = Path('/home/nathan/Expedition Competition')
PILOT = ROOT / 'Codex Work/ae_yolo_pilot'
HERE = PILOT / 'kaggle_ae_v2'
DATASET = PILOT / 'kaggle_fold0_v2_dataset'
OUT = PILOT / 'fold0_v2_kaggle'
CLASSES = ['kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha',
           'arkhitektury', 'fortifikatsii']
sys.path.insert(0, str(ROOT))
from koz3.sp_fuse import fuse


def main():
    raw_path = HERE / 'download' / 'ae_yolo_fold0_raw.csv'
    raw = pd.read_csv(raw_path)
    meta = {r['image']: r for r in json.loads((DATASET / 'heldout_metadata.json').read_text())}
    assert len(meta) == 298
    assert set(raw.image).issubset(meta)
    rows = []
    for image, detections in raw.groupby('image', sort=False):
        r = meta[image]
        T = Affine(*r['transform'])
        tr = Transformer.from_crs(r['crs'], 3857, always_xy=True)
        for d in detections.itertuples(index=False):
            cls = int(d.class_id)
            assert 0 <= cls < len(CLASSES)
            assert np.isfinite([d.confidence, d.x0, d.y0, d.x1, d.y1]).all()
            gx, gy = T * ((d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2)
            mx, my = tr.transform(gx, gy)
            rows.append((r['region'], CLASSES[cls], float(d.confidence), mx, my))
    points = pd.DataFrame(rows, columns=['region', 'cls', 'score', 'x', 'y'])
    OUT.mkdir(parents=True, exist_ok=True)
    points.to_parquet(OUT / 'oof_raw.parquet')
    merged = fuse([points], np.ones(1), match_m=4.0) if len(points) else points
    merged.to_parquet(OUT / 'oof_merged.parquet')
    (OUT / 'manifest.json').write_text((PILOT / 'fold0_v2/manifest.json').read_text())
    print('imported', len(points), 'raw ->', len(merged), 'merged')


if __name__ == '__main__':
    main()
