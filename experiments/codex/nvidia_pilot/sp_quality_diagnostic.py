"""Read-only site-level satellite quality diagnostic of the existing v9 OOF.

Diagnostic group definitions are unsupervised image percentiles. Scores are
descriptive and are never used to select a model or threshold on the same sites.
"""
import json
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path('/home/nathan/Expedition Competition')
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_assemble_oof import models, region_folds
from sp_fusion_oof import fc

OUT = ROOT / 'Codex Work/nvidia_pilot'


def main():
    ann = json.loads((ROOT / 'data/spor_prep/ann.json').read_text())
    site_spread = {}
    for c in ann:
        image = cv2.imread(str(ROOT / 'data/spor_prep/chips' / c['file']), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(c['file'])
        valid = image > 0
        if valid.sum() < 100:
            continue
        p2, p98 = np.percentile(image[valid], [2, 98])
        site_spread.setdefault(c['region'], []).append(float(p98-p2))
    quality = pd.Series({s: np.median(x) for s, x in site_spread.items()})
    rf = region_folds()
    quality = quality[quality.index.isin(rf)]
    cuts = np.quantile(quality, [1/3, 2/3])
    tier = pd.Series(np.where(quality < cuts[0], 'low',
                              np.where(quality < cuts[1], 'medium', 'high')),
                     index=quality.index)
    model = models(extra=('dfnodn',))
    pred = fuse(list(model.values()), np.ones(len(model)))
    gate = pd.read_parquet(ROOT / 'work/spor/oof_scene_cls.parquet')
    gate = gate[gate.p >= .1]
    g = pd.DataFrame({'region':gate.region, 'cls':'gorodishcha',
                      'score':gate.p, 'x':gate.x, 'y':gate.y3857})
    pred = pred[(pred.cls != 'gorodishcha') | pred.region.isin(set(gate.region))]
    pred = pd.concat([pred[pred.score >= .25], g], ignore_index=True)
    labels = gpd.read_parquet(ROOT / 'data/labels_3857.parquet')
    labels = labels[(labels['mod'] == 'SpOr') & labels.region.isin(quality.index)]
    report = {'cuts': list(map(float,cuts)), 'sites':len(quality), 'tiers':{},
              'fold_tiers':{}}
    for name in ('low', 'medium', 'high'):
        sites = set(tier[tier == name].index)
        p = pred[pred.region.isin(sites)]
        l = labels[labels.region.isin(sites)]
        overall, details = score(fc(p), to_gt_geojson(l))
        report['tiers'][name] = {
            'sites':len(sites), 'labels':len(l), 'predictions':len(p),
            'median_spread':float(quality[list(sites)].median()),
            'official_f1':float(overall),
            'classes':{k:{key:v[key] for key in ('f1','tp','fp','fn')}
                       for k,v in details.items() if k in (
                           'kurgany_povrezhdennye','kurgany_tselye','gorodishcha',
                           'arkhitektury','fortifikatsii')}}
        for fold in sorted(set(rf.values())):
            subset = {s for s in sites if rf[s] == fold}
            ps = pred[pred.region.isin(subset)]
            ls = labels[labels.region.isin(subset)]
            f1, _ = score(fc(ps), to_gt_geojson(ls))
            report['fold_tiers'][f'{fold}_{name}'] = {
                'sites':len(subset), 'labels':len(ls), 'f1':float(f1)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / 'sp_quality_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
