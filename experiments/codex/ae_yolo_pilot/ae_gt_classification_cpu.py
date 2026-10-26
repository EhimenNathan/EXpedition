"""CPU feasibility test: intact versus damaged kurgan texture at labelled centres.

Uses training GT only for fitting each site-disjoint fold. The held-out fold's
labels are used solely for the final AUC/balanced-accuracy report.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

ROOT=Path('/home/nathan/Expedition Competition')
HERE=ROOT/'Codex Work/ae_yolo_pilot'
sys.path.insert(0,str(HERE))
from ae_radial_rerank import image,crop_features

K=('kurgany_povrezhdennye','kurgany_tselye')
OUT=HERE/'radial_pilot/gt_class_features.parquet'


def build():
    ann=json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    bysite=defaultdict(list)
    folds={}
    for r in ann:
        bysite[r['region']].append(r)
        folds[r['region']]=r['fold']
    labels=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    labels=labels[(labels['mod']=='Ae')&labels.region.isin(bysite)&labels.class_name.isin(K)]
    transforms={}
    out=[]
    for region,g in labels.groupby('region'):
        chips=bysite[region]
        crs=chips[0]['crs']
        tr=transforms.setdefault(crs,Transformer.from_crs(3857,crs,always_xy=True))
        for r in g.itertuples():
            c=r.geometry.representative_point()
            ux,uy=tr.transform(c.x,c.y)
            best=None
            for chip in chips:
                a,b,x0,d,e,y0=chip['transform']
                px=(ux-x0)/a;py=(uy-y0)/e
                if 0<=px<chip['w'] and 0<=py<chip['h']:
                    margin=min(px,py,chip['w']-px,chip['h']-py)
                    if best is None or margin>best[0]:best=(margin,chip,px,py,abs(a))
            if best is None:continue
            _,chip,px,py,gsd=best
            f=crop_features(image(chip['file']),px,py,gsd)
            if f is not None:out.append((region,folds[region],int(r.class_name==K[1]),*f))
    cols=['region','fold','intact']+[f'pixel_{i}' for i in range(len(out[0])-3)]
    D=pd.DataFrame(out,columns=cols);D.to_parquet(OUT,index=False)
    return D


def main():
    D=pd.read_parquet(OUT) if OUT.exists() else build()
    print('extract',len(D),'fold/class',D.groupby(['fold','intact']).size().to_dict(),flush=True)
    cols=[c for c in D if c.startswith('pixel_')]
    for hold in range(4):
        train=({1,2,3}-{hold}) if hold else {1,2,3}
        tr=D[D.fold.isin(train)];va=D[D.fold==hold]
        model=lgb.LGBMClassifier(n_estimators=80,num_leaves=7,max_depth=4,
            learning_rate=.04,min_child_samples=25,reg_lambda=5.,
            class_weight='balanced',verbose=-1,n_jobs=4,random_state=20261024)
        model.fit(tr[cols],tr.intact)
        p=model.predict_proba(va[cols])[:,1]
        print('heldout',hold,'n',len(va),'intact',sum(va.intact),
              'AUC',round(roc_auc_score(va.intact,p),4),
              'AP',round(average_precision_score(va.intact,p),4),
              'balanced_acc',round(balanced_accuracy_score(va.intact,p>=.5),4),flush=True)

if __name__=='__main__':main()
