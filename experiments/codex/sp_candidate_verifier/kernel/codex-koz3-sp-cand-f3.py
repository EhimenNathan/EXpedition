"""Fixed candidate-objectness pilot. Heldout labels are absent from input."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from safetensors.torch import load_file

torch.set_num_threads(2)
torch.manual_seed(20261025)
np.random.seed(20261025)
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
    def __init__(self, arr, labels=None, augment=False):
        self.arr=arr;self.labels=labels;self.augment=augment
    def __len__(self):return len(self.arr)
    def __getitem__(self,i):
        x=self.arr[i].astype(np.float32)/255
        if self.augment:
            if np.random.rand()<.5:x=x[:,:,::-1].copy()
            if np.random.rand()<.5:x=x[:,::-1,:].copy()
            x=np.rot90(x,np.random.randint(4),axes=(1,2)).copy()
            x=np.clip((x-.5)*np.random.uniform(.75,1.25)+.5+np.random.uniform(-.1,.1),0,1)
            if np.random.rand()<.25:x=np.repeat(x.mean(axis=0,keepdims=True),3,axis=0)
        x=torch.from_numpy((x-np.array([.485,.456,.406])[:,None,None])/np.array([.229,.224,.225])[:,None,None]).float()
        return (x,torch.tensor(float(self.labels[i]),dtype=torch.float32)) if self.labels is not None else x

train_y=z['train_y'].astype(int);site=z['train_site'].astype(int)
weight=np.zeros(len(train_y),np.float64)
for s in np.unique(site):
    for c in (0,1):
        ix=np.where((site==s)&(train_y==c))[0]
        if len(ix):weight[ix]=(.5 if c else .5)/len(ix)
sampler=WeightedRandomSampler(torch.from_numpy(weight),num_samples=6000,replacement=True,generator=torch.Generator().manual_seed(20261025))
train=DataLoader(Crops(z['train_img'],train_y,True),batch_size=32,sampler=sampler,num_workers=2,pin_memory=True)
test=DataLoader(Crops(z['test_img']),batch_size=64,shuffle=False,num_workers=2,pin_memory=True)
model=timm.create_model('convnext_tiny',pretrained=False,num_classes=1)
weights=list(INPUT.rglob('convnext_tiny.safetensors'))
assert weights,'pretrained ConvNeXt weights absent from mounted dataset'
ck=load_file(str(weights[0]),device='cpu')
ck={k:v for k,v in ck.items() if not k.startswith('head.fc.')}
print('pretrained',model.load_state_dict(ck,strict=False),flush=True)
model.cuda()
opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.02)
scaler=torch.amp.GradScaler('cuda')
for epoch in range(8):
    model.train();loss_sum=0
    for x,y in train:
        x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(x).flatten();loss=nn.functional.binary_cross_entropy_with_logits(logits,y)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        loss_sum+=float(loss.detach())
    print('epoch',epoch+1,'mean_loss',loss_sum/len(train),flush=True)
torch.save(model.cpu().state_dict(),'/kaggle/working/candidate_verifier_f3.pt')
model.cuda()
model.eval();pred=[]
with torch.no_grad():
    for x in test:
        with torch.autocast('cuda',dtype=torch.float16):p=model(x.cuda(non_blocking=True)).sigmoid().flatten()
        pred.extend(p.float().cpu().numpy().tolist())
pd.DataFrame({'row_id':z['test_row'],'p_obj':pred}).to_csv('/kaggle/working/candidate_verifier_f3.csv',index=False)
print('output rows',len(pred),flush=True)
