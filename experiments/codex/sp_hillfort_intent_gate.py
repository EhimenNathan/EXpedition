"""Conservative hillfort scene intent gate, four-fold exact scorer."""
"""Prospective fold-3 exact-score test of a safe satellite site-presence gate.

The cutoff is the minimum out-of-fold site probability among positive sites
on folds 0-2. Thus no positive training site is gated. Fold 3 is scored only
after fixing this cutoff; regions absent from the feature cache are retained.
"""
import json
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.nested import weighted,_add_all
from koz3.score import score
from sp_assemble_oof import models,region_folds
from sp_dual_oof import fuse_cls
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry

NAMES=('dfine','rtdetr','hm','dfnodn','balq','balq3')
K=('kurgany_povrezhdennye','kurgany_tselye')

def main():
    P=pd.read_parquet(HERE/'sp_region_presence_probe.parquet')
    cutoff=.9
    S0=pd.read_parquet(ROOT/"work/spor/oof_scene_cls.parquet")
    p_hill=S0.groupby("region").p.max()
    max_k=P.set_index("region").max_score
    drop={r for r in p_hill.index.intersection(max_k.index) if p_hill[r]>=.9 and max_k[r]<.5}
    print('cutoff fixed inner',cutoff,'sites dropped by fold',P[P.region.isin(drop)].groupby('fold').size().to_dict(),flush=True)
    M=models(extra=list(NAMES[3:]))
    regions=sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
    RF=region_folds()
    F=fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES])
    F=F[(F.cls.isin(K)&(F.score>=.25)) |
        ((F.cls=='arkhitektury')&(F.score>=.15)) |
        ((~F.cls.isin(K+('arkhitektury',)))&(F.score>=.25))].copy()
    geom=scene_geometry();keep=[]
    for r in F.itertuples():
        d=min([(np.hypot(r.x-cx,r.y-cy)/max(h,1),sm) for cx,cy,h,sm in geom.get(r.region,[])] or [(0.0,True)])
        keep.append(d[1] or d[0]<=.5 or r.score>=.35)
    F=F[np.array(keep)]
    S=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet')
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='SpOr')&L.region.isin(regions)]
    gt={f:to_gt_geojson(L[L.region.map(RF)==f]) for f in range(4)}

    def make(fold,use_gate):
        g=F[F.region.map(RF)==fold].copy()
        if use_gate:g=g[~(g.region.isin(drop)&g.cls.isin(K))]
        extra=[]
        for r in g[g.cls.isin(K)].itertuples():
            other=K[1] if r.cls==K[0] else K[0]
            if (r.s_kt if other==K[1] else r.s_kp)>=.4:
                extra.append((r.region,other,r.score*.99,r.x,r.y))
        g=g[['region','cls','score','x','y']]
        if extra:g=pd.concat([g,pd.DataFrame(extra,columns=g.columns)],ignore_index=True)
        return fc(gate(g,S[S.region.map(RF)==fold]))
    def counts(pred,target):return {c:(v['tp'],v['fp'],v['fn']) for c,v in score(pred,target)[1].items()}
    B={f:counts(make(f,False),gt[f]) for f in range(4)}
    C={f:counts(make(f,True),gt[f]) for f in range(4)}
    result={'cutoff':cutoff,'inner_baseline':weighted(_add_all(B[f] for f in (0,1,2))),
            'inner_gate':weighted(_add_all(C[f] for f in (0,1,2))),
            'fold3_baseline':weighted(B[3]),'fold3_gate':weighted(C[3]),
            'all_baseline':weighted(_add_all(B.values())),'all_gate':weighted(_add_all(C.values())),
            'dropped_sites_fold3':sorted(r for r in drop if RF.get(r)==3)}
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    (HERE/'sp_hillfort_intent_gate.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
