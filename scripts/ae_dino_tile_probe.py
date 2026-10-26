from pathlib import Path
import json,sys,cv2,numpy as np,pandas as pd,torch,timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
ROOT=Path('/home/nathan/Expedition Competition');OUT=ROOT/'Codex Work/ae_dino_tile_feats.npz'
A=pd.DataFrame(json.load(open(ROOT/'data/ae_prep/ann.json')))
# target any kurgan in the chip; fortification-only chips remain negative.
K={0,1}; y=np.array([any(int(c) in K for c in cs) for cs in A.classes],np.int8)
if OUT.exists(): X=np.load(OUT)['X']
else:
 m=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=True,num_classes=0).cuda().eval()
 cfg=timm.data.resolve_model_data_config(m); norm=timm.data.create_transform(**cfg,is_training=False)
 from PIL import Image
 z=[]
 with torch.inference_mode():
  for i in range(0,len(A),32):
   ims=[]
   for f in A.file.iloc[i:i+32]:
    im=cv2.imread(str(ROOT/'data/ae_prep/chips'/f))
    if im is None: raise FileNotFoundError(f)
    ims.append(norm(Image.fromarray(im[:,:,::-1])))
   with torch.autocast('cuda',dtype=torch.bfloat16): q=m(torch.stack(ims).cuda()).float().cpu().numpy()
   z.append(q);print(i+len(ims),'/',len(A),flush=True)
 X=np.concatenate(z);np.savez_compressed(OUT,X=X)
print('shape',X.shape,'positive',y.sum(), 'regions',A.region.nunique())
# strict predefined geographic folds. Train on other folds, test one.
p=np.zeros(len(A),float)
for f in sorted(A['fold'].unique()):
 tr=A['fold'].to_numpy()!=f;te=~tr
 mod=make_pipeline(StandardScaler(),LogisticRegression(C=.03,class_weight='balanced',max_iter=3000))
 mod.fit(X[tr],y[tr]);p[te]=mod.predict_proba(X[te])[:,1]
 print('fold',f,'n',te.sum(),'pos',y[te].sum(),'auc',roc_auc_score(y[te],p[te]) if len(set(y[te]))>1 else np.nan,'ap',average_precision_score(y[te],p[te]))
print('pooled auc',roc_auc_score(y,p),'ap',average_precision_score(y,p),'prior',y.mean())
# site/scene presence via max and top-k tile scores
D=A[['region','scene','fold']].copy();D['y']=y;D['p']=p
G=D.groupby(['region','scene','fold']).agg(y=('y','max'),pmax=('p','max'),pmean=('p','mean'),n=('p','size')).reset_index()
print('scene max auc',roc_auc_score(G.y,G.pmax),'ap',average_precision_score(G.y,G.pmax),'prior',G.y.mean())
A.assign(y=y,p=p).to_parquet(ROOT/'Codex Work/ae_dino_tile_oof.parquet',index=False)
