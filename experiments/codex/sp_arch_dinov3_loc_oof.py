#!/usr/bin/env python3
"""Exact geographic OOF test of a DINOv3-SAT architecture rescue.

The accepted v26 runtime already computes the same satellite tokens for its
hillfort localizer.  This test adds at most one architecture point per scene,
only when the cross-fitted architecture map is strong and the deployed six
detector fusion has no nearby architecture point.  Existing predictions are
never removed or moved.  Every setting is selected on the other folds.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
NAMES=("dfine","rtdetr","hm","dfnodn","balq","balq3")
CLS="arkhitektury"
BASE=(1.1,20.0,1.0)
SETTINGS=[BASE]+[(m,g,a) for m in (.50,.65,.75,.85,.90,.95,.975)
                for g in (8.0,20.0,40.0) for a in (.5,.75,1.0)]

def match_points(scenes, points):
    out={}
    for r in scenes.itertuples():
        d=points[points.region==r.region]
        if len(d):
            j=np.argmin(np.hypot(d.cx3857.values-r.x,d.cy3857.values-r.y3857))
            q=d.iloc[j]
            out[r.Index]=(float(q.x3857),float(q.y3857),float(q.map_max))
    return out

def rescue_rows(scenes, loc, deployed, setting):
    m,gap,alpha=setting; rows=[]
    for r in scenes.itertuples():
        q=loc.get(r.Index)
        if q is None or q[2]<m: continue
        x,y,p=q
        b=deployed[(deployed.region==r.region)&(deployed.cls==CLS)]
        if len(b) and np.hypot(b.x-x,b.y-y).min()<gap: continue
        rows.append((r.region,CLS,alpha*p,x,y))
    return pd.DataFrame(rows,columns=['region','cls','score','x','y'])

def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts,nested,weighted,_add_all
    from sp_assemble_oof import models,region_folds
    from sp_fusion_oof import fc
    from sp_rescore_oof import scene_geometry
    from sp_dual_oof import fuse_cls
    from robust_accept import site_counts,report

    M=models(extra=list(NAMES[3:]));regions=sorted(set.intersection(*[set(m.region) for m in M.values()]));RF=region_folds()
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=L[(L['mod']=='SpOr')&L.region.isin(regions)]
    S=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet');S=S[S.region.isin(regions)]
    D=pd.read_parquet(ROOT/'work/sp_dinov3_multiclass_localizer_probe.parquet');D=D[D.cls==CLS]
    loc=match_points(S,D)
    F0=fuse_cls([M[n][M[n].region.isin(regions)] for n in NAMES]).reset_index(drop=True)
    geom=scene_geometry();cache={}
    def base(regs):
        key=tuple(regs)
        if key not in cache:
            F=F0[F0.region.isin(regs)]
            F=F[F.score>=np.where(F.cls==CLS,.15,.25)];keep=[]
            for r in F.itertuples():
                d=min([(np.hypot(r.x-cx,r.y-cy)/max(h,1),sm) for cx,cy,h,sm in geom.get(r.region,[])] or [(0.,True)])
                keep.append(d[1] or d[0]<=.5 or r.score>=.35)
            F=F[np.asarray(keep,bool)]
            K=("kurgany_povrezhdennye","kurgany_tselye")
            ex=[(r.region,K[1] if r.cls==K[0] else K[0],r.score*.99,r.x,r.y) for r in F[F.cls.isin(K)].itertuples()
                if (r.s_kt if r.cls==K[0] else r.s_kp)>=.4]
            cache[key]=pd.concat([F[['region','cls','score','x','y']],pd.DataFrame(ex,columns=['region','cls','score','x','y'])],ignore_index=True)
        return cache[key]
    def pipeline(st,regs):
        B=base(regs);R=rescue_rows(S[S.region.isin(regs)],loc,B,st)
        return pd.concat([B,R],ignore_index=True)

    folds=sorted({RF[r] for r in regions});gt={f:to_gt_geojson(L[L.region.map(RF)==f]) for f in folds}
    C=fold_counts(lambda st,f:fc(pipeline(st,[r for r in regions if RF[r]==f])),SETTINGS,folds,gt)
    for st in SETTINGS:
        t=_add_all(C[st].values());a=t.get(CLS,(0,0,0))
        print(f'{st} score {weighted(t):.5f} arch {tuple(int(v) for v in a)} folds '+
              ' '.join(f'{weighted(C[st][f]):.4f}' for f in folds))
    honest,ch,opt,optst,_=nested(C);b=weighted(_add_all(C[BASE].values()))
    print(f'baseline {b:.5f} | nested {honest:.5f} ({honest-b:+.5f}) | best fixed {opt:.5f} @ {optst} | chosen {ch}')
    print('outer changes',' '.join(f'{weighted(C[ch[f]][f])-weighted(C[BASE][f]):+.5f}' for f in folds))
    # The deployment gate requires fold nonregression.  Apply that same rule
    # inside each outer split: a setting is eligible only if it does not hurt
    # any of the remaining training folds, then maximize their pooled score.
    rch={}
    for f in folds:
        tr=[q for q in folds if q!=f]
        eligible=[st for st in SETTINGS if all(weighted(C[st][q])>=weighted(C[BASE][q])-1e-12 for q in tr)]
        rch[f]=max(eligible,key=lambda st:weighted(_add_all(C[st][q] for q in tr)))
    rh=weighted(_add_all(C[rch[f]][f] for f in folds))
    print(f'robust nested {rh:.5f} ({rh-b:+.5f}) | chosen {rch}')
    print('robust outer changes',' '.join(f'{weighted(C[rch[f]][f])-weighted(C[BASE][f]):+.5f}' for f in folds))
    st=max(set(rch.values()),key=list(rch.values()).count)
    gtr={r:to_gt_geojson(L[L.region==r]) for r in regions}
    CS=site_counts(lambda s,r:fc(pipeline(s,[r])),(BASE,st),regions,gtr)
    report(f'DINOv3 architecture rescue {st}',CS[BASE],CS[st])

if __name__=='__main__':main()
