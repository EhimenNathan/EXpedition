"""Prospective geographic fold test of rule selected on exploratory fold 1.

Frozen rule: architecture-only C-RADIO score >= .7, add if no v9 same-class
candidate within 5 metres in EPSG:3857. No validation threshold sweep.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path('/home/nathan/Expedition Competition')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(HERE)]
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_assemble_oof import models,region_folds
from sp_fusion_oof import fc

FOLD=int(sys.argv[1]) if len(sys.argv)>1 else 2
assert FOLD in (2,3)
sites={s for s,f in region_folds().items() if f==FOLD}
L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
gt=to_gt_geojson(L[(L['mod']=='SpOr')&L.region.isin(sites)])
M=models(extra=['dfnodn'])
base=fuse(list(M.values()),np.ones(len(M)))
base=base[base.score>=.25]
S=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet')
base=base[~((base.cls=='gorodishcha')&base.region.isin(set(S[S.p<.1].region)))]
a=S[S.p>=.1]
base=pd.concat([base,pd.DataFrame({'region':a.region,'cls':'gorodishcha','score':a.p,
                                   'x':a.x,'y':a.y3857})],ignore_index=True)
base=base[base.region.isin(sites)][['region','cls','score','x','y']]
cand=pd.read_parquet(HERE/f'cradio_sp_f{FOLD}_hires_merged.parquet')
add=cand[(cand.region.isin(sites))&(cand.cls=='arkhitektury')&(cand.score>=.7)]
keep=[]
for site,g in add.groupby('region'):
    existing=base[(base.region==site)&(base.cls=='arkhitektury')]
    if len(existing):
        tree=cKDTree(existing[['x','y']].to_numpy())
        distances,_=tree.query(g[['x','y']].to_numpy())
        keep.extend(g.index[distances>5.0].tolist())
    else:keep.extend(g.index.tolist())
extra=add.loc[keep][['region','cls','score','x','y']]
trial=pd.concat([base,extra],ignore_index=True)
before,bd=score(fc(base),gt)
after,ad=score(fc(trial),gt)
result={'fold':FOLD,'baseline':before,'with_rule':after,'delta':after-before,
        'baseline_objects':len(base),'added_objects':len(extra),
        'added_sites':sorted(extra.region.unique().tolist()),
        'architecture_before':bd['arkhitektury'],
        'architecture_after':ad['arkhitektury'],
        'rule':'arkhitektury score >= .7; add if no v9 same-class candidate within 5m'}
(HERE/f'cradio_arch_rule_f{FOLD}_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
