"""Fixed-threshold geographic OOF audit of existing Ae experts, read-only."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path('/home/nathan/Expedition Competition')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from ae_det_oof import ae_folds
from sp_fusion_oof import fc
from koz3.labels import to_gt_geojson
from koz3.score import score

folds = ae_folds()
sites = set(folds)
labels = gpd.read_parquet(ROOT/'data/labels_3857.parquet')
labels = labels[(labels['mod']=='Ae') & labels.region.isin(sites)]
dam = pd.read_parquet(ROOT/'work/ae/oof_spdet_cands.parquet')
dam = dam[dam.region.isin(sites) & (dam.cls=='kurgany_povrezhdennye') & (dam.score>=.7)]
fort = pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet')
fort = fort[fort.region.isin(sites) & (fort.cls=='fortifikatsii') & (fort.score>=.4)]
variants = {'deployed': dam, 'fort_only': fort,
            'combined': pd.concat([dam, fort], ignore_index=True)}
result = {}
for key in ['all'] + [str(f) for f in sorted(set(folds.values()))]:
    region_subset = sites if key == 'all' else {r for r,v in folds.items() if v==int(key)}
    gt = to_gt_geojson(labels[labels.region.isin(region_subset)])
    result[key] = {'sites': len(region_subset), 'models': {}}
    for name, data in variants.items():
        pred = data[data.region.isin(region_subset)]
        value, details = score(fc(pred), gt)
        result[key]['models'][name] = {'weighted_f1': value, 'predictions': len(pred),
                                        'class_counts': {cls: {k:int(v[k]) for k in ('tp','fp','fn')}
                                                         for cls,v in details.items()}}
    print(key, {n:round(v['weighted_f1'],6) for n,v in result[key]['models'].items()})
(ROOT/'Codex Work/ae_yolo_pilot/fortification_fcos_audit.json').write_text(
    json.dumps(result,ensure_ascii=False,indent=2))
