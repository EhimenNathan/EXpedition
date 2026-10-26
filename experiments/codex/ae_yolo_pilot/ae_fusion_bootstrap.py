"""Paired, fold-stratified site bootstrap for the nested Ae fusion candidate."""
import json
import sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'Codex Work/ae_yolo_pilot')]
from ae_crossclass_cpu import append_hedge,INTACT,DAMAGED
from koz3.labels import to_gt_geojson
from koz3.score import score,WEIGHTS
from sp_fusion_oof import fc

CAP={0:100,1:100,2:50,3:100} # chosen on the other three folds

def weighted(counts):
    num=den=0.
    for c,w in WEIGHTS.items():
        tp,fp,fn=counts.get(c,(0,0,0))
        if tp+fn:
            num+=w*(2*tp/(2*tp+fn+.5*fp));den+=w
    return num/den if den else 0.

def main():
    raw=pd.read_parquet(ROOT/'Codex Work/ae_yolo_pilot/fusion_pilot/equal_candidates.parquet')
    base=pd.read_parquet(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet')
    folds={r['region']:r['fold'] for r in json.loads((ROOT/'data/ae_prep/ann.json').read_text())}
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='Ae')&L.region.isin(folds)]
    new=[]
    for fold,cap in CAP.items():
        d=raw[raw.fold==fold]
        x=d[d.score>=.18]
        h=append_hedge(d,DAMAGED,INTACT,.2,base_threshold=.18)
        x=pd.concat([x,h],ignore_index=True)
        new.append(x.sort_values('score',ascending=False).groupby(['region','cls'],sort=False).head(cap))
    N=pd.concat(new,ignore_index=True)
    stats={}
    for region,fold in folds.items():
        g=to_gt_geojson(L[L.region==region])
        site=[]
        for d in (base[(base.region==region)&(base.score>=.4)],N[N.region==region]):
            _,res=score(fc(d),g)
            site.append({c:np.array([v['tp'],v['fp'],v['fn']],int) for c,v in res.items()})
        stats[region]=site
    def total(regions,i):
        counts={c:np.zeros(3,int) for c in WEIGHTS}
        for r in regions:
            for c,x in stats[r][i].items():counts[c]+=x
        return weighted(counts)
    names=list(folds)
    print('pooled baseline',total(names,0),'nested_candidate',total(names,1),'delta',total(names,1)-total(names,0))
    rng=np.random.default_rng(20261024);diff=[]
    by_fold={f:[r for r in names if folds[r]==f] for f in CAP}
    for _ in range(1000):
        sample=[r for f,rs in by_fold.items() for r in rng.choice(rs,len(rs),replace=True)]
        diff.append(total(sample,1)-total(sample,0))
    q=np.quantile(diff,[.025,.5,.975])
    print('paired_fold_stratified_site_bootstrap_delta_95pct',q.tolist(),
          'Pr_gain',float(np.mean(np.array(diff)>0)))

if __name__=='__main__':main()
