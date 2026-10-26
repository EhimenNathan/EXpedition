"""CPU-only whole-Ae score for the nested cross-class/site-cap diagnostic.

Fold 0 is reserved for the independent aerial pilot. Each fold's cap was
selected using only the other two folds in ae_sitecap_cpu.py.
"""
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

ROOT = Path('/home/nathan/Expedition Competition')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'Codex Work/ae_yolo_pilot')]
from ae_crossclass_cpu import append_hedge, INTACT, DAMAGED
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc


CAPS = {
    1: {INTACT: 200, DAMAGED: 80},
    2: {INTACT: None, DAMAGED: 40},
    3: {INTACT: 40, DAMAGED: 80},
}


def main():
    D = pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet')
    D = D[D.fold.isin(CAPS)]
    folds = D[['region', 'fold']].drop_duplicates().set_index('region').fold.to_dict()
    labels = gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    labels = labels[(labels['mod'] == 'Ae') & labels.region.isin(folds)]
    full_base, full_new = [], []
    for fold, caps in CAPS.items():
        d = D[D.fold == fold]
        base = d[d.score >= .4]
        pieces = [base[~base.cls.isin([INTACT, DAMAGED])]]
        hedge = pd.concat([
            append_hedge(d, DAMAGED, INTACT, .4),
            append_hedge(d, INTACT, DAMAGED, .3),
        ], ignore_index=True)
        both = pd.concat([base, hedge], ignore_index=True)
        for cls, cap in caps.items():
            x = both[both.cls == cls].sort_values('score', ascending=False)
            if cap is not None:
                x = x.groupby('region', sort=False).head(cap)
            pieces.append(x)
        candidate = pd.concat(pieces, ignore_index=True)
        gt = to_gt_geojson(labels[labels.region.map(folds) == fold])
        b, _ = score(fc(base), gt)
        n, _ = score(fc(candidate), gt)
        print('fold', fold, 'baseline', round(b, 6), 'candidate', round(n, 6),
              'delta', round(n-b, 6), 'pred', len(base), len(candidate), flush=True)
        full_base.append(base); full_new.append(candidate)
    gt = to_gt_geojson(labels)
    b, br = score(fc(pd.concat(full_base, ignore_index=True)), gt)
    n, nr = score(fc(pd.concat(full_new, ignore_index=True)), gt)
    print('pooled baseline', round(b, 6), 'nested candidate', round(n, 6),
          'delta', round(n-b, 6))
    for cls in br:
        print(cls, 'base', round(br[cls]['f1'], 5), 'candidate', round(nr[cls]['f1'], 5),
              'base_counts', [br[cls][x] for x in ('tp','fp','fn')],
              'new_counts', [nr[cls][x] for x in ('tp','fp','fn')])


if __name__ == '__main__':
    main()
