"""Conservative DINOv3-SAT detector rescue over the deployed six-member satellite fusion.

The equal-weight seventh-member test dilutes strong deployed candidates.  This
test leaves the deployed output byte-for-byte unchanged and only adds a kurgan
when either (a) a subthreshold deployed cluster agrees spatially with DINO-SAT,
or (b) DINO-SAT is exceptionally confident and spatially unique.  Every choice,
including no rescue, is selected on the other geographic folds.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
sys.path[:0]=[ROOT,os.path.join(ROOT,"scripts")]
from sp_assemble_oof import models,region_folds
from sp_dual_oof import fuse_cls,K
from sp_fusion_oof import fc
from sp_fusion4_oof import gate
from sp_rescore_oof import scene_geometry


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts,nested,weighted,_add_all
    from robust_accept import site_counts,report
    names=["dfine","rtdetr","hm","dfnodn","balq","balq3"]
    M=models(extra=names[3:]); D=pd.read_parquet(os.path.join(ROOT,"work/spor/oof_det_dsat_cands.parquet"))
    regs=sorted(set.intersection(*[set(M[n].region) for n in names],set(D.region)))
    RF=region_folds(); folds=sorted({RF[r] for r in regs})
    L=gpd.read_parquet(os.path.join(ROOT,"data/labels_3857.parquet"));L=L[(L["mod"]=="SpOr")&L.region.isin(regs)]
    gt={f:to_gt_geojson(L[L.region.map(RF)==f]) for f in folds}
    S=pd.read_parquet(os.path.join(ROOT,"work/spor/oof_scene_cls.parquet")); geom=scene_geometry()
    F=fuse_cls([M[n][M[n].region.isin(regs)] for n in names])
    far=[]
    for r in F.itertuples():
        q=min([(np.hypot(r.x-cx,r.y-cy)/max(h,1),sm) for cx,cy,h,sm in geom.get(r.region,[])] or [(0.,True)])
        far.append((not q[1]) and q[0]>.5)
    F=F.assign(far=far)

    # Attach each DINO candidate to the closest deployed kurgan-family cluster.
    rows=[]
    for reg,d in D[D.cls.isin(K)].groupby("region"):
        b=F[(F.region==reg)&F.cls.isin(K)]
        if len(b):
            tree=cKDTree(b[["x","y"]]); dist,ix=tree.query(d[["x","y"]],k=1)
            bb=b.iloc[ix]
            for rr,dd,z in zip(d.itertuples(),dist,bb.itertuples()):
                rows.append((reg,rr.cls,rr.score,rr.x,rr.y,float(dd),z.score,z.x,z.y,z.cls,z.far))
        else:
            rows += [(reg,r.cls,r.score,r.x,r.y,np.inf,0.,r.x,r.y,r.cls,False) for r in d.itertuples()]
    A=pd.DataFrame(rows,columns=["region","cls","score","x","y","dist","base_score","base_x","base_y","base_cls","base_far"])

    def dedup(q):
        out=[]
        for reg,g in q.sort_values("score",ascending=False).groupby("region",sort=False):
            keep=[]; pts=[]
            for r in g.itertuples():
                if not pts or min(np.hypot(r.x-x,r.y-y) for x,y in pts)>8:
                    keep.append(r);pts.append((r.x,r.y))
            out.extend(keep)
        return out

    def build(st,regions):
        kind,bthr,dthr=st
        G=F[F.region.isin(regions)]
        G=G[(G.score>=np.where(G.cls=="arkhitektury",.15,.25))&(~G.far|(G.score>=.35))]
        ex=[(r.region,K[1] if r.cls==K[0] else K[0],r.score*.99,r.x,r.y) for r in G[G.cls.isin(K)].itertuples()
            if (r.s_kt if r.cls==K[0] else r.s_kp)>=.4]
        if kind!="none":
            q=A[(A.region.isin(regions))&(A.score>=dthr)]
            if kind in ("agree","both"):
                qa=q[(q.dist<=8)&(q.base_score>=bthr)&(q.base_score<.25)]
                # Use the established ensemble coordinate and class.
                qa=qa.assign(x=qa.base_x,y=qa.base_y,cls=qa.base_cls)
            else: qa=q.iloc[:0]
            if kind in ("unique","both"):
                qu=q[(q.dist>8)&((q.score>=dthr)&(~q.base_far|(q.score>=.65)))]
            else: qu=q.iloc[:0]
            for r in dedup(pd.concat([qa,qu],ignore_index=True)):
                ex.append((r.region,r.cls,.251,r.x,r.y))
        B=G[["region","cls","score","x","y"]]
        if ex: B=pd.concat([B,pd.DataFrame(ex,columns=B.columns)],ignore_index=True)
        return B

    SS=[("none",0.,9.)]
    SS += [("agree",b,d) for b in (.08,.12,.16,.20) for d in (.40,.45,.50,.55,.60)]
    SS += [("unique",0.,d) for d in (.55,.60,.65,.70)]
    SS += [("both",b,d) for b in (.12,.16,.20) for d in (.55,.60,.65)]
    C=fold_counts(lambda s,f:fc(gate(build(s,[r for r in regs if RF[r]==f]),S[S.region.map(RF)==f])),SS,folds,gt)
    h,ch,o,os_,_=nested(C)
    base=weighted(_add_all(C[("none",0.,9.)].values()))
    print("base",base,"nested",h,"oracle",o,os_,"chosen",ch)
    for f in folds:
        print("fold",f,"base",weighted(C[("none",0.,9.)][f]),"new",weighted(C[ch[f]][f]),"delta",weighted(C[ch[f]][f])-weighted(C[("none",0.,9.)][f]))
    gtr={r:to_gt_geojson(L[L.region==r]) for r in regs}
    chosen_mode=max(set(ch.values()),key=list(ch.values()).count)
    CS=site_counts(lambda s,r:fc(gate(build(s,[r]),S[S.region==r])),(("none",0.,9.),chosen_mode),regs,gtr)
    report(f"DINO-SAT rescue {chosen_mode}",CS[("none",0.,9.)],CS[chosen_mode])

if __name__=="__main__": main()
