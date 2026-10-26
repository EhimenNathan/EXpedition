#!/usr/bin/env python3
"""Validation-composition score effect of scene-aware Ae cap-100 emission."""
from pathlib import Path
import sys,pickle,geopandas as gpd,pandas as pd,numpy as np
R=Path('/home/nathan/Expedition Competition');sys.path[:0]=[str(R),str(R/'scripts'),str(R/'Codex Work')]
from ae_layer_oof import layer_gt
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.nested import weighted,_add_all
from sp_fusion_oof import fc
MOD={'SpOr':30,'Li':13,'Ae':4,'MagGr':2,'EtGr':1,'TerFl':1,'Or':1};DEP={'SpOr':1.0,'Li':.15,'Or':.45,'MagGr':None,'EtGr':None,'TerFl':None};KP='kurgany_povrezhdennye'
def main():
 C=pickle.load(open(R/'work/joint_thr_counts.pkl','rb'));D=pd.read_parquet(R/'Codex Work/ae_sheet_selfadapt_features.parquet');L=gpd.read_parquet(R/'data/labels_3857.parquet');A=L[(L['mod']=='Ae')&L.region.isin(D.region.unique())];A=layer_gt(A);AC={}
 for reg in sorted(D.region.unique()):
  gt=to_gt_geojson(A[A.region==reg]);q=D[D.region==reg].sort_values('score',ascending=False).groupby('scene',group_keys=False).head(100).copy();q['cls']=KP
  for name,x in [('off',q.iloc[:0]),('cap100',q)]:
   _,z=score(fc(x),gt);AC[(reg,name)]={c:(v['tp'],v['fp'],v['fn']) for c,v in z.items()}
 regs={m:sorted({k[1] for k in C if k[0]==m}) for m in MOD if m!='Ae'};regs['Ae']=sorted(D.region.unique());rng=np.random.default_rng(20261026);delta=[];base=[];on=[]
 for _ in range(2000):
  picks={m:rng.choice(regs[m],MOD[m],replace=len(regs[m])<MOD[m]) for m in MOD};co=[];cn=[]
  for m in MOD:
   for r in picks[m]:
    if m=='Ae':co.append(AC[(r,'off')]);cn.append(AC[(r,'cap100')])
    else:co.append(C[(m,r,DEP[m])]);cn.append(C[(m,r,DEP[m])])
  b=weighted(_add_all(co));n=weighted(_add_all(cn));base.append(b);on.append(n);delta.append(n-b)
 print('base',np.mean(base),'cap100',np.mean(on),'delta mean',np.mean(delta),'q05/50/95',np.quantile(delta,[.05,.5,.95]),'P+',np.mean(np.array(delta)>0))
 # Per geographic fold on Ae only.
 fold=D.groupby('region').fold.first().astype(int).to_dict()
 for f in sorted(set(fold.values())):
  r=[x for x in regs['Ae'] if fold[x]==f];a=weighted(_add_all(AC[(x,'off')] for x in r));b=weighted(_add_all(AC[(x,'cap100')] for x in r));print('fold',f,len(r),a,b,b-a)
if __name__=='__main__':main()
