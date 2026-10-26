"""Audit satellite site folds for true-geodesic 25 km spatial overlap.

This is read-only and does not alter the existing fold assignment or models.
Future training may exclude the listed training regions per held-out fold.
"""
import json
from pathlib import Path

import geopandas as gpd
from pyproj import Geod

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ROOT / 'kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/folds.json'
LABELS = ROOT / 'data/labels_3857.parquet'
OUT = Path(__file__).with_suffix('.json')


def main():
    fold = {k.rsplit('__', 1)[0]: int(v) for k, v in json.loads(FOLDS.read_text()).items()}
    labels = gpd.read_parquet(LABELS)
    labels = labels[(labels['mod'] == 'SpOr') & labels.region.isin(fold)].to_crs(4326)
    points = {region: [(g.centroid.x, g.centroid.y) for g in group.geometry]
              for region, group in labels.groupby('region')}
    count = labels.region.value_counts().to_dict()
    geod = Geod(ellps='WGS84')
    regions = sorted(points)
    near = []
    for i, r in enumerate(regions):
        for s in regions[i + 1:]:
            if fold[r] == fold[s]:
                continue
            distance = min(geod.inv(lon, lat, lon2, lat2)[2] for lon, lat in points[r]
                           for lon2, lat2 in points[s])
            if distance < 25_000:
                near.append({'site_a': r, 'site_b': s, 'fold_a': fold[r], 'fold_b': fold[s],
                             'nearest_label_km': distance / 1000,
                             'labels_a': count[r], 'labels_b': count[s]})
    masks = {}
    for f in sorted(set(fold.values())):
        held = {r for r in regions if fold[r] == f}
        excluded = {p['site_b'] if p['fold_a'] == f else p['site_a']
                    for p in near if f in (p['fold_a'], p['fold_b'])}
        masks[str(f)] = {'heldout_sites': len(held),
                         'train_sites_before': len(regions) - len(held),
                         'nearby_train_sites_to_exclude': sorted(excluded),
                         'train_sites_after': len(regions) - len(held) - len(excluded),
                         'training_labels_excluded': sum(count[r] for r in excluded)}
    result = {'threshold_true_km': 25, 'regions_with_labels': len(regions),
              'labeled_objects': len(labels),
              'cross_fold_near_pairs': sorted(near, key=lambda p: p['nearest_label_km']),
              'fold_buffer_masks': masks}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
