#!/usr/bin/env python3
"""Select conservative scene-aware Ae policy by full-system validation-shaped utility."""
from pathlib import Path
import sys,pickle,geopandas as gpd,pandas as pd,numpy as np
R=Path('/home/nathan/Expedition Competition');sys.path[:0]=[str(R),str(R/'scripts'),str(R/'Codex Work')]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.nested import weighted,_add_all
from sp_fusion_oof import fc
MOD={'SpOr':30,'Li':13,'Ae':4,'MagGr':2,'EtGr':1,'TerFl':1,'Or':1};DEP={'SpOr':1.0,'Li':.15,'Or':.45,'MagGr':None,'EtGr':None,'TerFl':None};KP='kurgany_povrezhdennye';SET=[('off',)]+[(c,t) for c in (1,3,5,10,20,50,100) for t in (0.,.3,.4,.5,.6)]
def main():
 C=pickle.load(open(R/'work/joint_thr_counts.pkl','rb'));D=pd.read_parquet(R/'Codex Work/ae_sheet_selfadapt_features.parquet');L=gpd.read_parquet(R/'data/labels_3857.parquet');A=layer_gt(L[(L['mod']=='Ae')&L.region.isin(D.region.unique())]);regs={m:sorted({k[1] for k in C if k[0]==m}) for m in MOD if m!='Ae'};regs['Ae']=sorted(D.region.unique());fold=D.groupby('region').fold.first().astype(int).to_dict();AC={}
 for reg in regs['Ae']:
  gt=to_gt_geojson(A[A.region==reg]);g=D[D.region==reg].sort_values('score',ascending=False)
  for st in SET:
   if st==('off',):q=g.iloc[:0]
   else:
    cap,t=st;q=g[g.score>=t].groupby('scene',group_keys=False).head(cap).copy();q['cls']=KP
   _,z=score(fc(q),gt);AC[(reg,st)]={c:(v['tp'],v['fp'],v['fn']) for c,v in z.items()}
 rng=np.random.default_rng(20261026);draw=[]
 for _ in range(1000):
  fixed=[]
  for m in MOD:
   pick=rng.choice(regs[m],MOD[m],replace=len(regs[m])<MOD[m])
   if m=='Ae':draw.append(tuple(pick))
   else:fixed.extend(C[(m,r,DEP[m])] for r in pick)
  draw[-1]=(draw[-1],_add_all(fixed))
 def ev(st,allowed=None):
  out=[]
  for pick,fixed in draw:
   if allowed is not None:pick=tuple(x for x in pick if fold[x] in allowed)
   out.append(weighted(_add_all([fixed]+[AC[(r,st)] for r in pick])))
  return np.array(out)
 base=ev(('off',));rank=sorted(((float((ev(s)-base).mean()),s,float(np.mean(ev(s)>base)),np.quantile(ev(s)-base,[.05,.5,.95])) for s in SET),reverse=True,key=lambda x:x[0])
 for x in rank[:15]:print(x)
 # Fold-honest: select setting on regions from other folds using draws restricted to those Ae folds, test delta with target fold Ae counts injected into fixed public backgrounds.
 total=[];choices=[]
 for f in sorted(set(fold.values())):
  train=set(fold.values())-{f};best=max(SET,key=lambda s:(ev(s,train)-ev(('off',),train)).mean());choices.append((f,best))
  # Site-level exact full-system delta averaged over public backgrounds, using only heldout-fold Ae sites sampled to four.
  held=[r for r in regs['Ae'] if fold[r]==f];vals=[]
  for pick,fixed in draw:
   hp=np.random.default_rng(abs(hash((f,pick)))%(2**32)).choice(held,4,replace=len(held)<4)
   vals.append(weighted(_add_all([fixed]+[AC[(r,best)] for r in hp]))-weighted(_add_all([fixed]+[AC[(r,('off',))] for r in hp])))
  total.extend(vals);print('held',f,best,np.mean(vals),np.quantile(vals,[.05,.5,.95]))
 print('nested choices',choices,'mean',np.mean(total),'P+',np.mean(np.array(total)>0))
if __name__=='__main__':main()
