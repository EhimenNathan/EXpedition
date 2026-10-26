"""Site-disjoint optical verifier for dense Ae proposals.

The detector supplies localization candidates.  This model sees a 48 m film
context around each point and learns object/background rejection on source
sites only.  Heldout labels are saved solely for evaluation and are never read
by the training function.
"""
import argparse,json,os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from pyproj import Transformer
from rasterio.transform import Affine

ROOT=Path('/home/nathan/Expedition Competition');CW=ROOT/'Codex Work';S=96;H=S//2

def labels_for(cands):
 import geopandas as gpd
 import shapely
 L=gpd.read_parquet(ROOT/'data/labels_3857.parquet');L=L[(L['mod']=='Ae')&L.class_name.str.startswith('kurgany')]
 y=np.zeros(len(cands),np.uint8)
 for r,q in cands.groupby('region'):
  g=L[L.region==r]
  if len(g)==0:continue
  u=shapely.union_all(g.geometry.to_numpy());pts=shapely.points(q.x.to_numpy(),q.y.to_numpy())
  lat=2*np.arctan(np.exp(q.y.mean()/6378137))-np.pi/2;k=1/np.cos(lat)
  y[q.index]=shapely.distance(pts,u)<=4*k
 return y

def attach_chips(q,ann):
 q=q.copy();q['file']='';q['px']=-1;q['py']=-1;margin=np.full(len(q),-1.)
 for c in ann:
  T=Affine(*c['transform']);tr=Transformer.from_crs(3857,c['crs'],always_xy=True)
  X,Y=tr.transform(q.x.to_numpy(),q.y.to_numpy());ix,iy=(~T)*(X,Y);ix=np.asarray(ix);iy=np.asarray(iy)
  m=np.minimum.reduce([ix,iy,c['w']-1-ix,c['h']-1-iy]);use=(m>=H)&(m>margin)
  if use.any():
   jj=np.where(use)[0];q.iloc[jj,q.columns.get_loc('file')]=c['file'];q.iloc[jj,q.columns.get_loc('px')]=np.rint(ix[use]).astype(int);q.iloc[jj,q.columns.get_loc('py')]=np.rint(iy[use]).astype(int);margin[use]=m[use]
 return q[q.file!='']

def prepare(fold):
 ann=json.loads((ROOT/'data/ae_prep/ann.json').read_text());by={}
 for c in ann:by.setdefault(c['region'],[]).append(c)
 train_tag='train'+''.join(str(x) for x in range(4) if x!=fold)
 tr=pd.read_parquet(CW/f'ae_dense_p2_f{fold}_cands_{train_tag}.parquet').reset_index(drop=True)
 te=pd.read_parquet(CW/f'ae_dense_p2_f{fold}_cands.parquet').reset_index(drop=True)
 # Bound disk/RAM and balance geography before patch extraction.
 rng=np.random.default_rng(20261026);parts=[]
 for r,g in tr.groupby('region'):
  parts.append(g if len(g)<=1400 else g.iloc[rng.choice(len(g),1400,False)])
 tr=pd.concat(parts,ignore_index=True);te=te.reset_index(names='row_id')
 both=pd.concat([tr.assign(split=0,row_id=-1),te.assign(split=1)],ignore_index=True);both['target']=labels_for(both)
 imgs=[];ys=[];sites=[];rows=[];splits=[]
 chipdir=ROOT/'data/ae_prep/chips';cache={}
 for n,(r,q0) in enumerate(both.groupby('region')):
  q=attach_chips(q0,by[r])
  for p in q.itertuples():
   im=cache.get(p.file)
   if im is None:
    im=cv2.imread(str(chipdir/p.file),cv2.IMREAD_GRAYSCALE)
    if len(cache)<250:cache[p.file]=im
   z=im[p.py-H:p.py+H,p.px-H:p.px+H]
   if z.shape!=(S,S):continue
   imgs.append(z);ys.append(p.target);sites.append(r);rows.append(p.row_id);splits.append(p.split)
  if (n+1)%10==0:print('prepare sites',n+1,'patches',len(imgs),flush=True)
 imgs=np.stack(imgs);ys=np.asarray(ys,np.uint8);splits=np.asarray(splits,np.uint8);sites=np.asarray(sites)
 codes,names=pd.factorize(sites[splits==0]);out=CW/f'ae_p2_verifier_f{fold}.npz'
 np.savez(out,train_img=imgs[splits==0],train_y=ys[splits==0],train_site=codes.astype(np.int16),
          test_img=imgs[splits==1],test_y=ys[splits==1],test_row=np.asarray(rows)[splits==1])
 print(out,'train',sum(splits==0),'pos',ys[splits==0].sum(),'sites',len(names),'test',sum(splits==1),'pos',ys[splits==1].sum(),flush=True)

class Crops(Dataset):
 def __init__(self,x,y=None,aug=False,robust=False):self.x=x;self.y=y;self.aug=aug;self.robust=robust
 def __len__(self):return len(self.x)
 def __getitem__(self,i):
  x=self.x[i].astype(np.float32)/255
  if self.robust:
   lo,hi=np.quantile(x,[.05,.95]);x=np.clip((x-lo)/max(float(hi-lo),.05),0,1)
  if self.aug:
   x=np.rot90(x,np.random.randint(4)).copy()
   if np.random.rand()<.5:x=x[:,::-1].copy()
   x=np.clip((x-.5)*np.random.uniform(.65,1.35)+.5+np.random.uniform(-.12,.12),0,1)
   if np.random.rand()<.3:x=cv2.GaussianBlur(x,(3,3),0)
  x=np.repeat(x[None],3,0);x=(x-np.array([.485,.456,.406])[:,None,None])/np.array([.229,.224,.225])[:,None,None]
  t=torch.from_numpy(x).float();return (t,torch.tensor(float(self.y[i]))) if self.y is not None else t

def train(fold,robust=False):
 import timm
 z=np.load(CW/f'ae_p2_verifier_f{fold}.npz');y=z['train_y'].astype(int);site=z['train_site'];w=np.zeros(len(y),np.float64)
 for s in np.unique(site):
  for c in (0,1):
   ix=np.where((site==s)&(y==c))[0]
   if len(ix):w[ix]=.5/len(ix)
 sam=WeightedRandomSampler(torch.from_numpy(w),12000,True,generator=torch.Generator().manual_seed(20261026))
 dl=DataLoader(Crops(z['train_img'],y,True,robust),batch_size=48,sampler=sam,num_workers=4,pin_memory=True)
 te=DataLoader(Crops(z['test_img'],robust=robust),batch_size=96,num_workers=4,pin_memory=True)
 m=timm.create_model('convnext_tiny',pretrained=True,num_classes=1).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=.03);sc=torch.amp.GradScaler('cuda')
 for e in range(10):
  m.train();tot=0
  for x,t in dl:
   x=x.cuda(non_blocking=True);t=t.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=nn.functional.binary_cross_entropy_with_logits(m(x).flatten(),t)
   sc.scale(loss).backward();sc.step(opt);sc.update();tot+=float(loss.detach())
  print('verifier epoch',e+1,tot/len(dl),flush=True)
 m.eval();pred=[]
 with torch.no_grad():
  for x in te:
   with torch.autocast('cuda',dtype=torch.float16):p=m(x.cuda()).sigmoid().flatten()
   pred.extend(p.float().cpu().numpy())
 suffix='_robust' if robust else ''
 out=CW/f'ae_p2_verifier_f{fold}{suffix}.csv';pd.DataFrame({'row_id':z['test_row'],'p_obj':pred,'y':z['test_y']}).to_csv(out,index=False);print(out,len(pred),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,default=3);ap.add_argument('--prepare',action='store_true');ap.add_argument('--train',action='store_true');ap.add_argument('--robust',action='store_true');a=ap.parse_args()
 if a.prepare:prepare(a.fold)
 if a.train:train(a.fold,a.robust)
