"""How much is the intact/damaged kurgan confusion worth? Upper bound on LiDAR OOF predictions (official metric).

    python scripts/li_kurgan_oracle.py [--tag A2_ref_unet_r34_all]

Oracle: every kurgan prediction (either class) whose representative point lies in a kurgan label takes that label's
class; others keep theirs. The gap between oracle and actual = the most a perfect intact/damaged classifier could add.
Also reports the 'kurgan-any' detection quality (both classes pooled) and the actual score at the same settings."""
import argparse, os, sys
import pandas as pd
from shapely import STRtree
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
import official_rescore as R

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="A2_ref_unet_r34_all"); a = ap.parse_args()
P, G = R.load(a.tag)
K = ("kurgany_tselye", "kurgany_povrezhdennye")
P = P.copy(); P["orig"] = P.lcls
for s in P.group.unique():
    gk = G[(G.group == s) & G.lcls.isin(K)]
    if not len(gk):
        continue
    tree = STRtree(list(gk.geometry)); cls = list(gk.lcls)
    idx = P.index[(P.group == s) & P.lcls.isin(K)]
    for i in idx:
        hit = tree.query(P.at[i, "geometry"].representative_point(), predicate="intersects")
        if len(hit):
            P.at[i, "lcls"] = cls[hit[0]]
print("kurgan predictions relabelled:", int((P.lcls != P.orig).sum()), "of", int(P.orig.isin(K).sum()))
for m in sorted(P.thr.unique()):
    for c in (0.5, 0.55, 0.6):
        Pa = P[P.thr == m].assign(lcls=lambda d: d.orig)
        fa, ra = R.score(Pa, G, c); fo, ro = R.score(P[P.thr == m], G, c)
        Pk = Pa.assign(lcls=Pa.lcls.where(~Pa.lcls.isin(K), "kurgany_tselye")); Gk = G.assign(lcls=G.lcls.where(~G.lcls.isin(K), "kurgany_tselye"))
        _, rk = R.score(Pk, Gk, c)
        f = lambda r, k: f"{r.get(k, {}).get('F1', 0):.3f}"
        print(f"mask {m} conf>={c}: actual {fa:.4f} (tsel {f(ra, K[0])} povr {f(ra, K[1])})  oracle {fo:.4f} "
              f"(tsel {f(ro, K[0])} povr {f(ro, K[1])})  kurgan-any F1 {f(rk, K[0])}")
