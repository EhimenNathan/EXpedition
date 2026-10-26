"""Nested selection of cross-scene multiplicity hedges for satellite OOF.

Start from deployed-style region-pooled fusion.  Add one extra prediction only
where scene-preserved fusion independently supports the same class/location on
at least two source images.  This represents the organiser-confirmed duplicate
reference without globally duplicating detections.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("/home/nathan/Expedition Competition")
CW = ROOT / "Codex Work"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(CW)]
from sp_scene_multiplicity_oof import crs_map, load_layer_gt, stats, K


def additions(a, b, radius=8.0):
    rows = []
    for (region, cls), ga in a.groupby(["region", "cls"]):
        gb = b[(b.region == region) & (b.cls == cls)]
        if len(gb) < 2: continue
        tree = cKDTree(gb[["x", "y"]].to_numpy())
        used = set()
        for ra in ga.sort_values("score", ascending=False).itertuples():
            jj = tree.query_ball_point([ra.x, ra.y], radius)
            q = gb.iloc[jj].sort_values("score", ascending=False)
            q = q.drop_duplicates("scene")
            if q.scene.nunique() < 2: continue
            # Do not reuse the same second-scene evidence for nearby pooled rows.
            rb = q.iloc[1]
            key = (region, cls, rb.scene, round(rb.x, 1), round(rb.y, 1))
            if key in used: continue
            used.add(key)
            rows.append({"region": region, "scene": rb.scene, "cls": cls,
                         "score": float(rb.score), "ratio": float(rb.score / max(ra.score, 1e-6)),
                         "x": float(rb.x), "y": float(rb.y)})
    return pd.DataFrame(rows)


def f1(v):
    tp, fp, fn = v
    return 2 * tp / max(2 * tp + fn + .5 * fp, 1e-12)


def main():
    a = pd.read_parquet(CW / "sp_region_pooled_five_oof.parquet")
    b = pd.read_parquet(CW / "sp_scene_preserved_five_oof.parquet")
    gt = load_layer_gt(crs_map())
    fold_of = gt.groupby("region").fold.first().astype(int).to_dict()
    gt = gt[gt.region.isin(fold_of)].copy(); folds = sorted(set(fold_of.values()))
    add = additions(a, b)
    print("supported additions", len(add)); print(add.groupby("cls").size())
    settings = [(c, t, r) for c in K for t in (.25, .3, .35, .4, .45, .5, .6, .7) for r in (0, .5, .7, .85)]
    _, base = stats(a, gt, fold_of)
    print("base", {c: tuple(map(int, base[c])) for c in K})
    # Per class, select t/r on other folds and evaluate held-out fold. Recompute
    # fold counts because pooled totals alone cannot support nesting.
    from sp_scene_multiplicity_oof import fc
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    fold_counts = {}
    for sett in settings:
        c,t,r=sett; q=add[(add.cls==c)&(add.score>=t)&(add.ratio>=r)]
        p=pd.concat([a,q[a.columns.intersection(q.columns)]],ignore_index=True)
        for f in folds:
            pp=p[p.region.map(fold_of)==f]; gg=gt[gt.fold==f]
            _,res=score(fc(pp),to_gt_geojson(gg),classes=[c]);v=res[c]
            fold_counts[(sett,f)]=np.array((v['tp'],v['fp'],v['fn']),int)
    total={c:np.zeros(3,int) for c in K}; choices={c:[] for c in K}
    for c in K:
        ss=[s for s in settings if s[0]==c]
        for f in folds:
            best=max(ss,key=lambda s:f1(sum((fold_counts[(s,x)] for x in folds if x!=f),np.zeros(3,int))))
            choices[c].append(best);total[c]+=fold_counts[(best,f)]
    print("nested")
    for c in K:print(c,f1(total[c]),tuple(total[c]),choices[c])
    print("oracle top")
    for c in K:
        ss=[s for s in settings if s[0]==c]
        for s in sorted(ss,key=lambda z:f1(sum((fold_counts[(z,x)] for x in folds),np.zeros(3,int))),reverse=True)[:8]:
            v=sum((fold_counts[(s,x)] for x in folds),np.zeros(3,int));print(s,f1(v),tuple(v))
    add.to_parquet(CW / "sp_cross_scene_supported_additions.parquet")

if __name__ == '__main__':main()
