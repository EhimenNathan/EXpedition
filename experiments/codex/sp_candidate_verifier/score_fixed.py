"""Fixed-rule geographic-fold scorer for the candidate verifier.

Rescue only low-score kurgan candidates with p_obj >= .98, assigning .251
before the deployed threshold. Separately audit p_obj < .005 pruning.
Thresholds fixed before downloading heldout predictions.
"""
import json
import argparse
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CW=HERE.parent
ROOT=CW.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(CW)]
from koz3.labels import to_gt_geojson
from koz3.nested import weighted
from koz3.score import score
from sp_assemble_oof import models,region_folds
from sp_dual_oof import fuse_cls
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry

NAMES=('dfine','rtdetr','hm','dfnodn','balq','balq3')
K=('kurgany_povrezhdennye','kurgany_tselye')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pred',default=str(HERE/'download/candidate_verifier_f3.csv'))
    ap.add_argument('--fold',type=int,default=3)
    ap.add_argument('--hillfort-gate',action='store_true')
    a=ap.parse_args()
    P=pd.read_csv(a.pred)
    D=pd.read_parquet(CW/'sp_morph_features.parquet').reset_index(names='row_id')
    D=D.merge(P,on='row_id',how='left',validate='one_to_one')
    assert D[D.fold==a.fold].p_obj.notna().sum()==len(P)
    M=models(extra=list(NAMES[3:]))
    regions=sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
    RF=region_folds()
    F=fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES])
    # Exact candidate keys are produced by the same frozen fusion call.
    d=D[D.fold==a.fold][['region','cls','x','y','p_obj']]
    F=F.merge(d,on=['region','cls','x','y'],how='left',validate='one_to_one')
    F=F[F.region.map(RF)==a.fold].copy()
    S=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet')
    if a.hillfort_gate:
        p_hill=S.groupby('region').p.max()
        max_k=F[F.cls.isin(K)].groupby('region').score.max()
        drop={r for r in p_hill.index.intersection(max_k.index) if p_hill[r]>=.9 and max_k[r]<.5}
        F=F[~(F.cls.isin(K)&F.region.isin(drop))].copy()
        print('hillfort intent gate sites',len(drop),flush=True)
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='SpOr')&(L.region.map(RF)==a.fold)]
    GT=to_gt_geojson(L)
    geom=scene_geometry()
    def run(mode):
        f=F.copy()
        if mode=='rescue':
            sel=f.cls.isin(K)&f.score.between(.08,.25,inclusive='left')&(f.p_obj>=.98)
            f.loc[sel,'score']=.251
        elif mode=='prune':
            f=f[~(f.cls.isin(K)&(f.score>=.25)&(f.p_obj<.005))].copy()
        elif mode=='both':
            sel=f.cls.isin(K)&f.score.between(.08,.25,inclusive='left')&(f.p_obj>=.98)
            f.loc[sel,'score']=.251
            f=f[~(f.cls.isin(K)&(f.score>=.25)&(f.p_obj<.005))].copy()
        f=f[((f.cls.isin(K))&(f.score>=.25))|
            ((f.cls=='arkhitektury')&(f.score>=.15))|
            ((~f.cls.isin(K+('arkhitektury',)))&(f.score>=.25))].copy()
        keep=[]
        for r in f.itertuples():
            d0=min([(np.hypot(r.x-cx,r.y-cy)/max(h,1),sm) for cx,cy,h,sm in geom.get(r.region,[])] or [(0.0,True)])
            keep.append(d0[1] or d0[0]<=.5 or r.score>=.35)
        f=f[np.array(keep)]
        extra=[]
        for r in f[f.cls.isin(K)].itertuples():
            other=K[1] if r.cls==K[0] else K[0]
            if (r.s_kt if other==K[1] else r.s_kp)>=.4:
                extra.append((r.region,other,r.score*.99,r.x,r.y))
        f=f[['region','cls','score','x','y']]
        if extra:f=pd.concat([f,pd.DataFrame(extra,columns=f.columns)],ignore_index=True)
        pred=fc(gate(f,S[S.region.map(RF)==a.fold]))
        sc,by=score(pred,GT)
        return {'score':weighted({k:(v['tp'],v['fp'],v['fn']) for k,v in by.items()}),
                'kurgans':{k:{z:v[z] for z in ('tp','fp','fn')} for k,v in by.items() if k in K},
                'pred_rows':len(pred['features'])}
    out={m:run(m) for m in ('baseline','rescue','prune','both')}
    print(json.dumps(out,indent=2,ensure_ascii=False))
    out_path=HERE/('score_fixed_'+Path(a.pred).stem+f'_fold{a.fold}'+('_hill' if a.hillfort_gate else '')+'.json')
    out_path.write_text(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
