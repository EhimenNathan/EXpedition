"""Read-only CPU feasibility audit for learned LiDAR/Ae spatial alignment.

Measures the residual displacement of existing co-annotated training sites.
This is descriptive only: missing Ae labels and true historical changes prevent
unmatched LiDAR objects from being called false positives.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path('/home/nathan/Expedition Competition')
CLASSES = {'kurgany_povrezhdennye', 'kurgany_tselye'}


def main():
    labels = gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    labels = labels[labels.class_name.isin(CLASSES) & labels['mod'].isin(['Li', 'Ae'])]
    sites = sorted(set(labels[labels['mod']=='Li'].region) & set(labels[labels['mod']=='Ae'].region))
    all_d, all_same, n_li, n_near, n_same = [], [], 0, 0, 0
    translation_tests = []
    per_site = {}
    for site in sites:
        li = labels[(labels.region==site) & (labels['mod']=='Li')]
        ae = labels[(labels.region==site) & (labels['mod']=='Ae')]
        if not len(ae) or not len(li): continue
        xy = np.array([(g.representative_point().x, g.representative_point().y) for g in ae.geometry])
        li_xy = np.array([(g.representative_point().x, g.representative_point().y) for g in li.geometry])
        tree = cKDTree(xy)
        dist, idx = tree.query(li_xy)
        lat_rad = 2*np.arctan(np.exp(np.median(xy[:,1])/6378137.0))-np.pi/2
        true_d = dist*np.cos(lat_rad)
        same = li.class_name.to_numpy() == ae.iloc[idx].class_name.to_numpy()
        near = true_d <= 10
        _, back = cKDTree(li_xy).query(xy)
        mutual = back[idx] == np.arange(len(li_xy))
        pairs = np.where(near & same & mutual)[0]
        if len(pairs) >= 6:
            # Deterministic spatially interleaved split: labels used to infer
            # the offset cannot also validate it.
            pairs = pairs[np.argsort(li_xy[pairs, 0] + li_xy[pairs, 1])]
            fit, test = pairs[::2], pairs[1::2]
            delta = np.median(xy[idx[fit]]-li_xy[fit], axis=0)
            before = np.linalg.norm((xy[idx[test]]-li_xy[test]), axis=1)*np.cos(lat_rad)
            after = np.linalg.norm((xy[idx[test]]-li_xy[test]-delta), axis=1)*np.cos(lat_rad)
            translation_tests.append({'site':site,'fit':len(fit),'test':len(test),
                                      'offset_true_m':float(np.linalg.norm(delta)*np.cos(lat_rad)),
                                      'before_median_m':float(np.median(before)),
                                      'after_median_m':float(np.median(after)),
                                      'before_p90_m':float(np.quantile(before,.9)),
                                      'after_p90_m':float(np.quantile(after,.9))})
        all_d.extend(true_d[near].tolist()); all_same.extend(same[near].tolist())
        n_li += len(li); n_near += int(near.sum()); n_same += int((near & same).sum())
        per_site[site] = {'li': len(li), 'ae': len(ae), 'near_10m': int(near.sum()),
                          'same_class_near_10m': int((near & same).sum()),
                          'matched_median_m': float(np.median(true_d[near])) if near.any() else None}
    out = {'sites': len(per_site), 'li_objects': n_li, 'near_any_ae_10m': n_near,
           'near_same_class_10m': n_same,
           'distance_quantiles_m_for_near': {str(q): float(np.quantile(all_d,q)) for q in [.1,.5,.9,.99]},
           'same_class_rate_within_10m': float(np.mean(all_same)), 'per_site': per_site,
           'translation_tests': translation_tests,
           'translation_sites_improved_median': sum(r['after_median_m']<r['before_median_m'] for r in translation_tests)}
    path = ROOT/'Codex Work/ae_yolo_pilot/li_ae_alignment_cpu.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='per_site'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
