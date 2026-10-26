"""Exploratory fold-0 fusion audit for the independent SegFormer satellite pilot.

All comparisons use the fixed deployed six-member decoding pipeline.  This file
only identifies a candidate rule; any rule must be frozen and tested on a newly
trained prospective geographic fold before it can count as evidence.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path('/home/nathan/Expedition Competition')
HERE = ROOT / 'Codex Work/nvidia_pilot'
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]

from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_assemble_oof import models, region_folds
from sp_dual_oof import fuse_cls
from sp_fusion4_oof import gate
from sp_fusion_oof import fc
from sp_rescore_oof import scene_geometry

NAMES = ('dfine','rtdetr','hm','dfnodn','balq','balq3')
K = ('kurgany_povrezhdennye','kurgany_tselye')


def postprocess(fused, sites, scene):
    g = fused[fused.region.isin(sites)].copy()
    geom = scene_geometry()
    far = []
    for r in g.itertuples():
        d = min([(np.hypot(r.x-cx, r.y-cy)/max(h,1), small)
                 for cx,cy,h,small in geom.get(r.region, [])] or [(0., True)])
        far.append((not d[1]) and d[0] > .5)
    g['far'] = far
    base = np.where(g.cls == 'arkhitektury', .15, .25)
    g = g[(g.score >= base) & (~g.far | (g.score >= .35))]
    extra = [(r.region, K[1] if r.cls == K[0] else K[0], r.score*.99, r.x, r.y)
             for r in g[g.cls.isin(K)].itertuples()
             if (r.s_kt if r.cls == K[0] else r.s_kp) >= .4]
    g = pd.concat([g[['region','cls','score','x','y']],
                   pd.DataFrame(extra, columns=['region','cls','score','x','y'])],
                  ignore_index=True)
    return gate(g, scene[scene.region.isin(sites)])


def add_novel(base, cand, threshold, classes):
    add = cand[(cand.score >= threshold) & cand.cls.isin(classes)].copy()
    keep = []
    for (site, cls), g in add.groupby(['region','cls']):
        old = base[(base.region == site) & (base.cls == cls)]
        if len(old):
            dist, _ = cKDTree(old[['x','y']].to_numpy()).query(g[['x','y']].to_numpy())
            keep.extend(g.index[dist > 4.].tolist())
        else:
            keep.extend(g.index.tolist())
    return pd.concat([base, add.loc[keep, ['region','cls','score','x','y']]], ignore_index=True), len(keep)


def main(fold=0, member_weights=(.25,.5,1.,2.)):
    rf = region_folds()
    sites = {s for s,f in rf.items() if f == fold}
    labels = gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    gt = to_gt_geojson(labels[(labels['mod']=='SpOr') & labels.region.isin(sites)])
    mm = models(extra=['dfnodn','balq','balq3'])
    scene = pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet')
    cand = pd.read_parquet(HERE/f'segformer_sp_f{fold}_merged.parquet')
    base_fused = fuse_cls([mm[n] for n in NAMES])
    base = postprocess(base_fused, sites, scene)
    bscore, _ = score(fc(base), gt)
    print('baseline', round(bscore,6), 'objects',len(base), flush=True)

    for w in member_weights:
        # fuse_cls uses equal members, so scale this member's scores to express
        # its relative vote while preserving the deployed member calibration.
        z = cand.copy(); z['score'] = (z.score*w).clip(upper=1.)
        trial = postprocess(fuse_cls([*[mm[n] for n in NAMES], z]), sites, scene)
        s,_ = score(fc(trial),gt)
        print('member_weight',w,'score',round(s,6),'delta',round(s-bscore,6),'n',len(trial),flush=True)

    groups = {'all':list(K)+['gorodishcha','arkhitektury','fortifikatsii'],
              'kurgans':list(K), 'damaged':[K[0]], 'intact':[K[1]],
              'fort':['fortifikatsii'], 'arch':['arkhitektury']}
    for name, classes in groups.items():
        for t in (.3,.4,.5,.6,.7,.8):
            trial,n = add_novel(base,cand,t,classes)
            s,_ = score(fc(trial),gt)
            print('rescue',name,t,'score',round(s,6),'delta',round(s-bscore,6),'add',n,flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv)>1 else 0,
         (float(sys.argv[2]),) if len(sys.argv)>2 else (.25,.5,1.,2.))
