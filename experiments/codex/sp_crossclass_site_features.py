"""Label-free cross-class satellite site descriptors from geographic OOF detections."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sp_assemble_oof import models
from sp_dual_oof import fuse_cls

NAMES=('dfine','rtdetr','hm','dfnodn','balq','balq3')
CLASSES=('gorodishcha','fortifikatsii','arkhitektury')

def main():
    M=models(extra=list(NAMES[3:]))
    regions=sorted(set.intersection(*(set(M[n].region) for n in NAMES)))
    F=fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES])
    S=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet')
    rows=[]
    for region in regions:
        q=F[F.region==region]
        d={'region':region}
        for c in CLASSES:
            sc=q[q.cls==c].score.to_numpy()
            d[f'x_{c}_max']=float(sc.max()) if len(sc) else 0
            d[f'x_{c}_n25']=float(np.log1p((sc>=.25).sum()))
            d[f'x_{c}_n40']=float(np.log1p((sc>=.4).sum()))
        z=S[S.region==region]
        for c in ('p','prob','score','p_hillfort'):
            if c in z:
                d['x_scene_hillfort_max']=float(z[c].max());break
        rows.append(d)
    O=pd.DataFrame(rows).fillna(0)
    O.to_parquet(HERE/'sp_crossclass_site_features.parquet',index=False)
    print('saved',len(O),'regions',O.columns.tolist(),flush=True)

if __name__=='__main__':main()
