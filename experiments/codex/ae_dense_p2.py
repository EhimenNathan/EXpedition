"""High-resolution, site-balanced Ae point detector.

This is an independent ablation of ae_dense_heatmap.py.  A full-resolution
convolutional detail path is fused with the semantic U-Net output so objects
smaller than the ConvNeXt stride-4 stem are still directly visible.  Sampling
is uniform by geographic site and includes 40% random background crops.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from ae_dense_heatmap import DenseSampler, MEAN, STD, ROOT, build as build_old
from koz3 import hm
from koz3.sp_train import CLS, load_chips, prep_dir


class SiteBalancedSampler(DenseSampler):
 def __init__(self,chips,seed):
  super().__init__(chips,seed)
  self.by_site={}
  for i,c in enumerate(chips): self.by_site.setdefault(c['region'],[]).append(i)
  self.sites=sorted(self.by_site)
  self.site_chip_w={}
  for r,ii in self.by_site.items():
   w=np.sqrt(np.array([1+len(self.kobjs[i]) for i in ii],float));self.site_chip_w[r]=w/w.sum()
 def crop(self):
  r=self.sites[int(self.rng.integers(len(self.sites)))];ii=self.by_site[r]
  i=int(self.rng.choice(ii,p=self.site_chip_w[r]));c=self.chips[i];objs=self.kobjs[i];im=self.img(c);h,w=im.shape[:2]
  if objs and self.rng.random()<.6:
   p=objs[self.rng.integers(len(objs))][1].representative_point();cx=p.x+self.rng.normal(0,96);cy=p.y+self.rng.normal(0,96)
  else:cx=self.rng.uniform(0,w);cy=self.rng.uniform(0,h)
  from ae_dense_heatmap import CROP,KCLS
  x0=float(np.clip(cx-CROP/2,min(0,w-CROP),max(0,w-CROP)));y0=float(np.clip(cy-CROP/2,min(0,h-CROP),max(0,h-CROP)))
  M=np.array([[1,0,-x0],[0,1,-y0]],np.float32);out=cv2.warpAffine(np.ascontiguousarray(im),M,(CROP,CROP),borderValue=0)
  from shapely.affinity import affine_transform
  oo=[(KCLS[cl],affine_transform(g,[1,0,0,1,-x0,-y0])) for cl,g in objs]
  ch,_=hm.draw_targets((CROP,CROP),oo,2,sig_frac=0,sig_min=1.5,sig_max=1.5);target=np.concatenate([np.maximum(ch[0],ch[1])[None],ch])
  g=out.astype(np.float32).mean(2);g=g*self.rng.uniform(.72,1.32)+self.rng.uniform(-28,28)
  if self.rng.random()<.45:g=255*np.clip(g/255,0,1)**self.rng.uniform(.65,1.5)
  if self.rng.random()<.25:g=cv2.GaussianBlur(g,(int(self.rng.choice([3,5])),)*2,0)
  if self.rng.random()<.3:g+=self.rng.normal(0,self.rng.uniform(2,9),g.shape)
  val=(out.max(2)>0).astype(np.float32);img=np.repeat(np.clip(g,0,255)[None],3,0);img=(img-MEAN)/STD*val
  k=int(self.rng.integers(4));fl=self.rng.random()<.5
  img=np.rot90(img,k,axes=(-2,-1));target=np.rot90(target,k,axes=(-2,-1));val=np.rot90(val,k)
  if fl:img=img[...,::-1];target=target[...,::-1];val=val[...,::-1]
  return [np.ascontiguousarray(x,np.float32) for x in (img,target,val)]


class P2Net(nn.Module):
 def __init__(self,pretrained=True):
  super().__init__()
  import segmentation_models_pytorch as smp
  self.context=smp.Unet('tu-convnext_tiny',encoder_weights='imagenet' if pretrained else None,
    in_channels=3,classes=24,decoder_channels=(256,128,64,32,24))
  self.detail=nn.Sequential(
    nn.Conv2d(3,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.GELU(),
    nn.Conv2d(16,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.GELU())
  self.head=nn.Sequential(nn.Conv2d(40,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,3,1))
 def forward(self,x):return self.head(torch.cat([self.context(x),self.detail(x)],1))


def build(pretrained=True):return P2Net(pretrained)


def train(fold,iters,bs):
 chips=[c for c in load_chips('Ae') if c['fold']!=fold];s=SiteBalancedSampler(chips,20261026+fold);torch.manual_seed(20261026+fold)
 m=build().cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01);sch=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,iters,pct_start=.05)
 ex=ThreadPoolExecutor(6);f=[ex.submit(s.batch,bs) for _ in range(3)];t0=time.time()
 for it in range(iters):
  img,t,v=[x.cuda(non_blocking=True) for x in f.pop(0).result()];f.append(ex.submit(s.batch,bs))
  with torch.autocast('cuda',dtype=torch.bfloat16):o=m(img.to(memory_format=torch.channels_last))
  w=v[:,None];loss=hm.focal_hm(o[:,:1],t[:,:1],w)+.4*hm.focal_hm(o[:,1:],t[:,1:],w)
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();sch.step()
  if it%250==0 or it==iters-1:print('p2 fold',fold,'it',it,'loss',float(loss),round(time.time()-t0),flush=True)
 torch.save({k:v.half() for k,v in m.state_dict().items()},f'/home/nathan/Expedition Competition/Codex Work/ae_dense_p2_f{fold}.pt')


@torch.no_grad()
def infer(fold,thr=.02,data_folds=None,tag=''):
 from pyproj import Transformer
 from rasterio.transform import Affine
 from koz3.sp_fuse import fuse
 m=build(False).cuda().eval();m.load_state_dict({k:v.float() for k,v in torch.load(f'/home/nathan/Expedition Competition/Codex Work/ae_dense_p2_f{fold}.pt',map_location='cpu').items()})
 rows=[];data_folds={fold} if data_folds is None else set(data_folds);chips=[c for c in load_chips('Ae') if c['fold'] in data_folds]
 for n,c in enumerate(chips,1):
  rgb=cv2.imread(os.path.join(prep_dir('Ae'),'chips',c['file']))[:,:,::-1];v=rgb.max(2)>0;g=rgb.astype(np.float32).mean(2)
  x=np.repeat(g[None],3,0);x=(x-MEAN)/STD*v[None]
  # Match the 512-pixel training context and use 31% overlap.
  p=hm.predict_tiled(m,x.astype(np.float32),3,tile=512,overlap=160).astype(np.float32)
  T=Affine(*c['transform']);tr=Transformer.from_crs(c['crs'],3857,always_xy=True)
  for _,xx,yy,s in hm.decode(p[:1],v,thr=thr,nms=5):
   yi,xi=int(yy),int(xx);y0,y1=max(0,yi-2),yi+3;x0,x1=max(0,xi-2),xi+3
   cl=int(np.argmax([p[1,y0:y1,x0:x1].mean(),p[2,y0:y1,x0:x1].mean()]));X,Y=tr.transform(*(T*(xx,yy)))
   rows.append((c['region'],CLS[cl],float(s),X,Y,c['fold']))
  if n%25==0:print('p2 infer',n,len(chips),len(rows),flush=True)
 d=pd.DataFrame(rows,columns=['region','cls','score','x','y','fold'])
 f=fuse([d[['region','cls','score','x','y']]],[1.0],match_m=3.0)
 # Recover actual data fold after fusing by the region mapping.
 rf={c['region']:c['fold'] for c in chips};f['fold']=f.region.map(rf)
 suffix=f'_{tag}' if tag else ''
 out=f'/home/nathan/Expedition Competition/Codex Work/ae_dense_p2_f{fold}_cands{suffix}.parquet';f.to_parquet(out);print(out,len(d),len(f),flush=True)


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--iters',type=int,default=5000);ap.add_argument('--bs',type=int,default=4);ap.add_argument('--train',action='store_true');ap.add_argument('--infer',action='store_true');ap.add_argument('--data-folds',default='');ap.add_argument('--tag',default='');a=ap.parse_args()
 if a.train:train(a.fold,a.iters,a.bs)
 if a.infer:infer(a.fold,data_folds=[int(x) for x in a.data_folds.split(',') if x] or None,tag=a.tag)
