"""Fixed Ae D-FINE-scale addition to the existing FCOS+joint OOF fusion.

Single-fold results are exploratory. The candidate must be trained/evaluated
on additional geographic folds before changing a submission.
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(Path(__file__).resolve().parent)]
from koz3.sp_fuse import fuse
from koz3.labels import to_gt_geojson
from koz3.score import score
from sp_fusion_oof import fc
from ae_crossclass_cpu import append_hedge,DAMAGED,INTACT


def model_input(path, sites, threshold):
    d=pd.read_parquet(path)
    return d[d.region.isin(sites)&(d.score>=threshold)][['region','cls','score','x','y']]


def make(sources):
    m=fuse(sources,np.ones(len(sources)),match_m=4.0)
    base=m[m.score>=.18]
    hedge=append_hedge(m,DAMAGED,INTACT,.2,base_threshold=.18)
    return (pd.concat([base,hedge],ignore_index=True)
            .sort_values('score',ascending=False)
            .groupby(['region','cls'],sort=False).head(100))


def evaluate(d,gt):
    val,res=score(fc(d),gt)
    return {'ae_f1':float(val),'objects':len(d),
            'classes':{c:{k:float(v[k]) if k=='f1' else int(v[k]) for k in ('f1','tp','fp','fn')}
                       for c,v in res.items()}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True,choices=range(4))
    ap.add_argument('--scale-pred',required=True)
    ap.add_argument('--scale-threshold',type=float,default=.4)
    ap.add_argument('--all-classes',action='store_true');a=ap.parse_args()
    ann=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    sites={r['region'] for r in ann if r['fold']==a.fold}
    sources=[model_input(ROOT/'work/ae'/f,sites,.2) for f in
             ('oof_ae_fcos_fast_cands.parquet','oof_ae_dfj_cands.parquet')]
    scale=model_input(a.scale_pred,sites,a.scale_threshold)
    source_full=pd.read_parquet(a.scale_pred)
    assert set(source_full.fold.unique())=={a.fold}
    assert set(source_full.region).issubset(sites)
    labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    labels=labels[(labels['mod']=='Ae')&labels.region.isin(sites)]
    gt=to_gt_geojson(labels)
    baseline=make(sources)
    addition=scale if a.all_classes else scale[scale.cls==DAMAGED]
    fused=make(sources+[addition])
    result={'fold':a.fold,'heldout_sites':len(sites),'scale_threshold':a.scale_threshold,
            'base_threshold':.18,'hedge_threshold':.2,'site_class_cap':100,
            'baseline':evaluate(baseline,gt),'with_scale_damaged':evaluate(fused,gt)}
    result['site_deltas']={}
    for site in sorted(sites):
        site_gt=to_gt_geojson(labels[labels.region==site])
        before=evaluate(baseline[baseline.region==site],site_gt)
        after=evaluate(fused[fused.region==site],site_gt)
        result['site_deltas'][site]={
            'ae_f1_before':before['ae_f1'],'ae_f1_after':after['ae_f1'],
            'ae_f1_delta':after['ae_f1']-before['ae_f1'],
            'damaged_tp_delta':after['classes'][DAMAGED]['tp']-before['classes'][DAMAGED]['tp'],
            'damaged_fp_delta':after['classes'][DAMAGED]['fp']-before['classes'][DAMAGED]['fp']}
    out=Path(__file__).resolve().parent/f'ae_scale_fusion_fold{a.fold}.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('baseline',result['baseline']['ae_f1'],'with_scale',result['with_scale_damaged']['ae_f1'],
          'delta',result['with_scale_damaged']['ae_f1']-result['baseline']['ae_f1'],flush=True)
    print('saved',out,flush=True)


if __name__=='__main__':main()
