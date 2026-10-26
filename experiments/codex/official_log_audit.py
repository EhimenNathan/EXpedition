"""Paired validation-shaped CPU ablations prompted by the v9 official log.

All settings share the same draws; no model is trained or submission modified.
The simulator reuses existing geographic OOF predictions and official scorer.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT=Path('/home/nathan/Expedition Competition')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from koz3.labels import to_gt_geojson
from koz3.score import score
from val_sim import COMPOSITION,build_predictions


def main():
    P=build_predictions()
    labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    grouped={(m,r):g for (m,r),g in labels.groupby(['mod','region'])}
    pools={m:sorted(set(labels[labels['mod']==m].region)) for m in COMPOSITION}
    rng=np.random.default_rng(20261024)
    rows=[]
    for draw in range(40):
        picks=[]
        for m,n in COMPOSITION.items():
            picks += [(m,r) for r in rng.choice(pools[m],min(n,len(pools[m])),replace=len(pools[m])<n)]
        gt=to_gt_geojson(pd.concat([grouped[k] for k in picks]))
        baseline=[];no_ae=[];no_or=[];no_aerial=[]
        for m,r in picks:
            fs=P.get(m,{}).get(r,[])
            baseline+=fs
            if m!='Ae':no_ae+=fs
            if m!='Or':no_or+=fs
            if m not in ('Ae','Or'):no_aerial+=fs
        row={'draw':draw}
        for name,fs in [('baseline',baseline),('no_ae',no_ae),('no_or',no_or),('no_aerial',no_aerial)]:
            row[name]=score({'type':'FeatureCollection','features':fs},gt)[0]
        rows.append(row)
        if (draw+1)%10==0:print('draws',draw+1,flush=True)
    out=pd.DataFrame(rows)
    summary={name:{'mean':float(out[name].mean()),
                   'delta_from_baseline':float((out[name]-out.baseline).mean()),
                   'positive_draws':int((out[name]>out.baseline).sum())}
             for name in ('baseline','no_ae','no_or','no_aerial')}
    d=ROOT/'Codex Work';out.to_csv(d/'official_log_ablation_draws.csv',index=False)
    (d/'official_log_ablation_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
