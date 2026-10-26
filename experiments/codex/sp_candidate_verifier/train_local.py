"""Local fixed ConvNeXt candidate verifier; training never reads test_y."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import timm
from safetensors.torch import load_file

torch.set_num_threads(2);torch.manual_seed(20261025);np.random.seed(20261025)
class Crops(Dataset):
 def __init__(self,x,y=None,aug=False):self.x=x;self.y=y;self.aug=aug
 def __len__(self):return len(self.x)
 def __getitem__(self,i):
  x=self.x[i].astype(np.float32)/255
  if self.aug:
   if np.random.rand()<.5:x=x[:,:,::-1].copy()
   if np.random.rand()<.5:x=x[:,::-1,:].copy()
   x=np.rot90(x,np.random.randint(4),axes=(1,2)).copy();x=np.clip((x-.5)*np.random.uniform(.75,1.25)+.5+np.random.uniform(-.1,.1),0,1)
   if np.random.rand()<.25:x=np.repeat(x.mean(0,keepdims=True),3,0)
  x=torch.from_numpy((x-np.array([.485,.456,.406])[:,None,None])/np.array([.229,.224,.225])[:,None,None]).float()
  return (x,torch.tensor(float(self.y[i]))) if self.y is not None else x
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);a=ap.parse_args();here=Path(__file__).parent
 z=np.load(here/f'local_fold{a.fold}.npz');y=z['train_y'].astype(int);site=z['train_site'];w=np.zeros(len(y),np.float64)
 for s in np.unique(site):
  for c in (0,1):
   ix=np.where((site==s)&(y==c))[0]
   if len(ix):w[ix]=.5/len(ix)
 sam=WeightedRandomSampler(torch.from_numpy(w),6000,True,generator=torch.Generator().manual_seed(20261025))
 tr=DataLoader(Crops(z['train_img'],y,True),batch_size=32,sampler=sam,num_workers=2,pin_memory=True)
 te=DataLoader(Crops(z['test_img']),batch_size=64,num_workers=2,pin_memory=True)
 m=timm.create_model('convnext_tiny',pretrained=False,num_classes=1)
 ck=load_file(str(here/'upload/convnext_tiny.safetensors'),device='cpu');ck={k:v for k,v in ck.items() if not k.startswith('head.fc.')};m.load_state_dict(ck,strict=False);m.cuda()
 opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=.02);sc=torch.amp.GradScaler('cuda')
 for e in range(8):
  m.train();tot=0
  for x,t in tr:
   x=x.cuda(non_blocking=True);t=t.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=nn.functional.binary_cross_entropy_with_logits(m(x).flatten(),t)
   sc.scale(loss).backward();sc.step(opt);sc.update();tot+=float(loss.detach())
  print('epoch',e+1,tot/len(tr),flush=True)
 m.eval();pred=[]
 with torch.no_grad():
  for x in te:
   with torch.autocast('cuda',dtype=torch.float16):p=m(x.cuda()).sigmoid().flatten()
   pred.extend(p.float().cpu().numpy())
 out=here/f'candidate_verifier_f{a.fold}_local.csv';pd.DataFrame({'row_id':z['test_row'],'p_obj':pred}).to_csv(out,index=False)
 print(out,'rows',len(pred),flush=True)
if __name__=='__main__':main()
