"""Dense, kurgan-only Ae point detector with geographic holdout.

The model emits an any-kurgan center map and two class center maps.  It uses
small fixed-width point targets so touching cemetery mounds remain separable.
No heldout labels are used during training or checkpoint selection.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

ROOT="/home/nathan/Expedition Competition"
sys.path.insert(0,ROOT)
from koz3.sp_train import CLS,load_chips,prep_dir
from koz3 import hm

CROP=512
KCLS={0:0,1:1}
MEAN=np.array([.485,.456,.406],np.float32)[:,None,None]*255
STD=np.array([.229,.224,.225],np.float32)[:,None,None]*255

class DenseSampler:
 def __init__(self,chips,seed):
  self.chips=chips;self.rng=np.random.default_rng(seed);self.cache={};self.dir=os.path.join(prep_dir('Ae'),'chips')
  self.kobjs=[]
  for c in chips:self.kobjs.append([(cl,g) for cl,g in c['objs'] if cl in KCLS])
  self.w=np.array([1+len(x) for x in self.kobjs],float);self.w/=self.w.sum()
 def img(self,c):
  x=self.cache.get(c['file'])
  if x is None:
   x=cv2.imread(os.path.join(self.dir,c['file']))[:,:,::-1]
   if len(self.cache)<500:self.cache[c['file']]=x
  return x
 def crop(self):
  i=int(self.rng.choice(len(self.chips),p=self.w));c=self.chips[i];objs=self.kobjs[i];im=self.img(c);h,w=im.shape[:2]
  if objs and self.rng.random()<.8:
   p=objs[self.rng.integers(len(objs))][1].representative_point();cx=p.x+self.rng.normal(0,96);cy=p.y+self.rng.normal(0,96)
  else:cx=self.rng.uniform(0,w);cy=self.rng.uniform(0,h)
  x0=float(np.clip(cx-CROP/2,min(0,w-CROP),max(0,w-CROP)));y0=float(np.clip(cy-CROP/2,min(0,h-CROP),max(0,h-CROP)))
  M=np.array([[1,0,-x0],[0,1,-y0]],np.float32);out=cv2.warpAffine(np.ascontiguousarray(im),M,(CROP,CROP),borderValue=0)
  from shapely.affinity import affine_transform
  oo=[(KCLS[cl],affine_transform(g,[1,0,0,1,-x0,-y0])) for cl,g in objs]
  ch,_=hm.draw_targets((CROP,CROP),oo,2,sig_frac=0,sig_min=1.5,sig_max=1.5);anyhm=np.maximum(ch[0],ch[1])[None];target=np.concatenate([anyhm,ch])
  g=out.astype(np.float32).mean(2);g=g*self.rng.uniform(.72,1.32)+self.rng.uniform(-28,28)
  if self.rng.random()<.45:g=255*np.clip(g/255,0,1)**self.rng.uniform(.65,1.5)
  if self.rng.random()<.25:g=cv2.GaussianBlur(g,(int(self.rng.choice([3,5])),)*2,0)
  if self.rng.random()<.3:g+=self.rng.normal(0,self.rng.uniform(2,9),g.shape)
  val=(out.max(2)>0).astype(np.float32);img=np.repeat(np.clip(g,0,255)[None],3,0);img=(img-MEAN)/STD*val
  k=int(self.rng.integers(4));fl=self.rng.random()<.5
  img=np.rot90(img,k,axes=(-2,-1));target=np.rot90(target,k,axes=(-2,-1));val=np.rot90(val,k)
  if fl:img=img[...,::-1];target=target[...,::-1];val=val[...,::-1]
  return [np.ascontiguousarray(x,np.float32) for x in (img,target,val)]
 def batch(self,n):
  b=[self.crop() for _ in range(n)];return [torch.from_numpy(np.stack([x[j] for x in b])) for j in range(3)]

def build(pretrained=True):
 import segmentation_models_pytorch as smp
 return smp.Unet('tu-convnext_tiny',encoder_weights='imagenet' if pretrained else None,in_channels=3,classes=3,decoder_channels=(256,128,64,32,16))

def train(fold,iters,bs):
 chips=[c for c in load_chips('Ae') if c['fold']!=fold];s=DenseSampler(chips,20261025+fold);torch.manual_seed(20261025+fold)
 m=build().cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01);sch=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,iters,pct_start=.05)
 ex=ThreadPoolExecutor(6);f=[ex.submit(s.batch,bs) for _ in range(3)];t0=time.time()
 for it in range(iters):
  img,t,v=[x.cuda(non_blocking=True) for x in f.pop(0).result()];f.append(ex.submit(s.batch,bs))
  with torch.autocast('cuda',dtype=torch.bfloat16):o=m(img.to(memory_format=torch.channels_last))
  w=v[:,None];loss=hm.focal_hm(o[:,:1],t[:,:1],w)+.6*hm.focal_hm(o[:,1:],t[:,1:],w)
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();sch.step()
  if it%250==0 or it==iters-1:print('fold',fold,'it',it,'loss',float(loss),round(time.time()-t0),flush=True)
 torch.save({k:v.half() for k,v in m.state_dict().items()},f'/home/nathan/Expedition Competition/Codex Work/ae_dense_f{fold}.pt')

@torch.no_grad()
def infer(fold,thr=.02,tile=1024,overlap=192,tag=''):
 from pyproj import Transformer
 from rasterio.transform import Affine
 from shapely.geometry import Point
 from koz3.sp_train import dedupe
 m=build(False).cuda().eval();m.load_state_dict({k:v.float() for k,v in torch.load(f'/home/nathan/Expedition Competition/Codex Work/ae_dense_f{fold}.pt',map_location='cpu').items()})
 rows=[];chips=[c for c in load_chips('Ae') if c['fold']==fold]
 for n,c in enumerate(chips,1):
  rgb=cv2.imread(os.path.join(prep_dir('Ae'),'chips',c['file']))[:,:,::-1]
  v=rgb.max(2)>0;g=rgb.astype(np.float32).mean(2);x=np.repeat(g[None],3,0);x=(x-MEAN)/STD*v[None]
  p=hm.predict_tiled(m,x.astype(np.float32),3,tile=tile,overlap=overlap).astype(np.float32)
  T=Affine(*c['transform']);tr=Transformer.from_crs(c['crs'],3857,always_xy=True)
  for _,xx,yy,s in hm.decode(p[:1],v,thr=thr,nms=5):
   yi,xi=int(yy),int(xx);y0,y1=max(0,yi-2),yi+3;x0,x1=max(0,xi-2),xi+3;cl=int(np.argmax([p[1,y0:y1,x0:x1].mean(),p[2,y0:y1,x0:x1].mean()]))
   X,Y=tr.transform(*(T*(xx,yy)));rows.append((c['region'],CLS[cl],float(s),X,Y,fold))
  if n%25==0:print('infer',n,len(chips),len(rows),flush=True)
 d=pd.DataFrame(rows,columns=['region','cls','score','x','y','fold'])
 # Geographic-coordinate dedupe shared with other OOF candidate tables.
 from koz3.sp_fuse import fuse
 f=fuse([d[['region','cls','score','x','y']]],[1.0],match_m=3.0);f['fold']=fold
 suffix=f'_{tag}' if tag else ''
 out=f'/home/nathan/Expedition Competition/Codex Work/ae_dense_f{fold}_cands{suffix}.parquet';f.to_parquet(out);print(out,len(d),len(f),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--iters',type=int,default=5000);ap.add_argument('--bs',type=int,default=8);ap.add_argument('--train',action='store_true');ap.add_argument('--infer',action='store_true');ap.add_argument('--tile',type=int,default=1024);ap.add_argument('--overlap',type=int,default=192);ap.add_argument('--tag',default='');a=ap.parse_args()
 if a.train:train(a.fold,a.iters,a.bs)
 if a.infer:infer(a.fold,tile=a.tile,overlap=a.overlap,tag=a.tag)
