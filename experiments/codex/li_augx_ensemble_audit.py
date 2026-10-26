"""Read-only, fixed-setting geographic OOF comparison of LiDAR ensembles.

Uses existing held-out probability maps. No model training, threshold search,
or edits to Claude's artifacts. No fused map is saved to disk.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT=Path('/home/nathan/Expedition Competition')
THR=float(os.environ.get('KOZ3_LI_AUDIT_THR','0.15'))
sys.path.insert(0,str(ROOT))
from koz3.li_train import labels,site_items,decode_site_v2
from koz3.labels import to_gt_geojson
from koz3.score import score


def main():
    L=labels()
    items={it['region']:it for it in site_items(L)}
    with (ROOT/'work/li/oof_ens.pkl').open('rb') as f:old=pickle.load(f)
    with (ROOT/'work/li/oof_augx.pkl').open('rb') as f:new=pickle.load(f)
    assert set(old)==set(new)
    assert set(old)<=set(items)
    regions=set(old)
    gt=to_gt_geojson(L[L.region.isin(regions)])
    outputs={name:[] for name in ('seed_ensemble','augx','equal_blend','two_seed_one_augx')}
    by_fold={f:{name:[] for name in outputs} for f in sorted({items[r]['fold'] for r in regions})}
    for i,r in enumerate(sorted(regions),1):
        a=old[r].astype(np.float32)
        b=new[r].astype(np.float32)
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        it=items[r]
        meta=dict(it['meta'],region=r)
        for name,p in (('seed_ensemble',a),('augx',b),
                       ('equal_blend',.5*a+.5*b),('two_seed_one_augx',(2*a+b)/3)):
            fs=decode_site_v2(p,it['valid'],meta,thr=THR,dual=.05)
            outputs[name]+=fs
            by_fold[it['fold']][name]+=fs
        if i%10==0:print('decoded',i,'/',len(regions),flush=True)
    result={}
    for name,fs in outputs.items():
        s,details=score({'type':'FeatureCollection','features':fs},gt)
        result[name]={'weighted_f1':float(s),'objects':len(fs),
                      'class_f1':{k:float(v['f1']) for k,v in details.items()}}
        print(name,round(s,6),'objects',len(fs),flush=True)
    result['folds']={}
    for fold, variants in by_fold.items():
        fold_regions={r for r in regions if items[r]['fold']==fold}
        fold_gt=to_gt_geojson(L[L.region.isin(fold_regions)])
        result['folds'][str(fold)]={}
        for name,fs in variants.items():
            s,details=score({'type':'FeatureCollection','features':fs},fold_gt)
            result['folds'][str(fold)][name]={'weighted_f1':float(s),'objects':len(fs),
                'class_counts':{k:{key:int(v[key]) for key in ('tp','fp','fn')} for k,v in details.items()}}
        print('fold',fold,'sites',len(fold_regions),
              {k:round(v['weighted_f1'],6) for k,v in result['folds'][str(fold)].items()},flush=True)
    dest=ROOT/f'Codex Work/li_augx_ensemble_audit_thr{THR:.2f}.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('saved',dest,flush=True)


if __name__=='__main__':main()
