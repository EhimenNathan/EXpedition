"""Exploratory complement audit on fold 1; not a deploy threshold selection."""
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

sites={s for s,f in region_folds().items() if f==1}
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
cand=pd.read_parquet(HERE/'cradio_sp_f1_hires_merged.parquet')
cand=cand[cand.region.isin(sites)][['region','cls','score','x','y']]
baseline_score=score(fc(base),gt)[0]
print('baseline',baseline_score,len(base),flush=True)
for classes in [('all',None),('damaged',{'kurgany_povrezhdennye'}),
                ('architecture',{'arkhitektury'})]:
    name,allowed=classes
    for t in (.4,.5,.6,.7,.8):
        add=cand[cand.score>=t]
        if allowed is not None:add=add[add.cls.isin(allowed)]
        keep=[]
        for (site,cl),g in add.groupby(['region','cls']):
            existing=base[(base.region==site)&(base.cls==cl)]
            if len(existing):
                tree=cKDTree(existing[['x','y']].to_numpy())
                distances,_=tree.query(g[['x','y']].to_numpy())
                keep.extend(g.index[distances>5.0].tolist())
            else:keep.extend(g.index.tolist())
        extra=add.loc[keep]
        trial=pd.concat([base,extra],ignore_index=True)
        val=score(fc(trial),gt)[0]
        print(name,t,'add',len(extra),'F1',round(val,5),'delta',round(val-baseline_score,5),flush=True)
