"""Fixed diagnostics for heldout candidate verifier; no threshold selection."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--pred',default=str(HERE/'download/candidate_verifier_f3.csv'))
a=ap.parse_args()
F=pd.read_parquet(HERE.parent/'sp_morph_features.parquet').reset_index(names='row_id')
P=pd.read_csv(a.pred)
Q=F.merge(P,on='row_id',how='inner',validate='one_to_one')
assert set(Q.fold)=={3} and len(Q)==len(P)
K={'kurgany_povrezhdennye','kurgany_tselye'}
L=pd.read_parquet(HERE.parent.parent/'data/labels_3857.parquet')
k_sites=set(L[(L['mod']=='SpOr')&L.class_name.isin(K)].region)
Q['site_has_k']=Q.region.isin(k_sites)

out={'n':len(Q),'positives':int(Q.y_obj.sum())}
for name,q in [('all',Q),('k_sites',Q[Q.site_has_k]),('no_k_sites',Q[~Q.site_has_k]),
               ('baseline',Q[Q.score>=.25]),('rescue_pool',Q[Q.score.between(.08,.25,inclusive='left')])]:
 d={'n':len(q),'positive':int(q.y_obj.sum())}
 if len(q) and q.y_obj.nunique()==2:
  d['ap_score']=float(average_precision_score(q.y_obj,q.score))
  d['ap_verifier']=float(average_precision_score(q.y_obj,q.p_obj))
  d['auc_score']=float(roc_auc_score(q.y_obj,q.score))
  d['auc_verifier']=float(roc_auc_score(q.y_obj,q.p_obj))
 for th in (.005,.5,.9,.98):
  z=q[q.p_obj>=th];d[f'p{th}_n']=len(z);d[f'p{th}_positive']=int(z.y_obj.sum())
 out[name]=d
print(json.dumps(out,indent=2,ensure_ascii=False))
(HERE/('analyze_'+Path(a.pred).stem+'.json')).write_text(json.dumps(out,indent=2,ensure_ascii=False))
