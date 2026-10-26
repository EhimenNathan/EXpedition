"""Fixed site-disjoint SatlasPretrain aerial-backbone candidate verifier pilot."""
import os,subprocess,sys
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','satlaspretrain-models==0.3.1'],check=True)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import satlaspretrain_models as spm

torch.set_num_threads(2)
torch.manual_seed(20261026);np.random.seed(20261026)
INPUT=Path('/kaggle/input')
print('mounted input files',[(str(p),p.stat().st_size) for p in INPUT.rglob('*') if p.is_file()][:30],flush=True)
npz=list(INPUT.rglob('candidate_crops_f3.npz'))
if npz:
 z=np.load(npz[0])
else:
 train_img=list(INPUT.rglob('train_img.npy'))
 assert train_img,'candidate crops absent from mounted dataset'
 folder=train_img[0].parent
 z={k:np.load(folder/f'{k}.npy') for k in ('train_img','train_y','train_site','train_row','test_img','test_row')}
assert 'test_y' not in (z.files if hasattr(z,'files') else z.keys())

class Crops(Dataset):
 def __init__(self,x,y=None,aug=False):self.x=x;self.y=y;self.aug=aug
 def __len__(self):return len(self.x)
 def __getitem__(self,i):
  x=self.x[i].astype(np.float32)/255
  if self.aug:
   if np.random.rand()<.5:x=x[:,:,::-1].copy()
   if np.random.rand()<.5:x=x[:,::-1,:].copy()
   x=np.rot90(x,np.random.randint(4),axes=(1,2)).copy()
   x=np.clip((x-.5)*np.random.uniform(.8,1.2)+.5+np.random.uniform(-.08,.08),0,1)
   if np.random.rand()<.25:x=np.repeat(x.mean(axis=0,keepdims=True),3,axis=0)
  x=torch.from_numpy(x.copy()).float()
  return (x,torch.tensor(int(self.y[i]),dtype=torch.long)) if self.y is not None else x

y=z['train_y'].astype(int);site=z['train_site'].astype(int)
w=np.zeros(len(y),np.float64)
for s in np.unique(site):
 for c in (0,1):
  ii=np.where((site==s)&(y==c))[0]
  if len(ii):w[ii]=.5/len(ii)
sampler=WeightedRandomSampler(torch.from_numpy(w),num_samples=4000,replacement=True,generator=torch.Generator().manual_seed(20261026))
train=DataLoader(Crops(z['train_img'],y,True),batch_size=4,sampler=sampler,num_workers=2,pin_memory=True)
test=DataLoader(Crops(z['test_img']),batch_size=16,shuffle=False,num_workers=2,pin_memory=True)
model=spm.Weights().get_pretrained_model('Aerial_SwinB_SI',fpn=True,head=spm.Head.CLASSIFY,num_categories=2)
model.cuda()
opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.02)
scaler=torch.amp.GradScaler('cuda')
for epoch in range(4):
 model.train();total=0;steps=0;opt.zero_grad(set_to_none=True)
 for j,(x,t) in enumerate(train):
  x=x.cuda(non_blocking=True);t=t.cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.float16):
   _,loss=model(x,t);loss=loss/4
  if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss epoch={epoch} batch={j}')
  scaler.scale(loss).backward();total+=float(loss.detach())*4
  if (j+1)%4==0 or j+1==len(train):
   scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
   scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);steps+=1
 print('epoch',epoch+1,'loss',total/len(train),'opt_steps',steps,flush=True)
torch.save(model.cpu().state_dict(),'/kaggle/working/candidate_satlas_f3.pt')
model.cuda()
model.eval();pred=[]
with torch.no_grad():
 for x in test:
  with torch.autocast('cuda',dtype=torch.float16):prob,_=model(x.cuda(non_blocking=True))
  pred.extend(prob[:,1].float().cpu().numpy().tolist())
pd.DataFrame({'row_id':z['test_row'],'p_obj':pred}).to_csv('/kaggle/working/candidate_satlas_f3.csv',index=False)
print('rows',len(pred),flush=True)
