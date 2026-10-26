"""Frozen NVIDIA C-RADIOv3-B plus dense head, satellite geographic fold 0.

The foundation encoder is fixed. Only training-site crops fit the head; fold 0
is held out for one final object-level evaluation. No submission integration.
"""
import argparse
import contextlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from cradio_v3b.hf_model import RADIOConfig, RADIOModel
from train_sp_segformer import SatelliteCrops, MEAN, STD

HERE = Path(__file__).resolve().parent
BACKBONE = HERE/'cradio_v3b'


def load_backbone(device):
    config = RADIOConfig.from_pretrained(str(BACKBONE))
    model = RADIOModel(config)
    model.load_state_dict(load_file(str(BACKBONE/'model.safetensors')), strict=True)
    model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    return model


class DenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(768,256,1),nn.BatchNorm2d(256),nn.GELU(),
                                  nn.Conv2d(256,128,3,padding=1),nn.GELU(),
                                  nn.Conv2d(128,6,1))

    def forward(self, features,size):
        return F.interpolate(self.layers(features.float()),size=size,
                             mode='bilinear',align_corners=False)


class DenseHeadHiRes(nn.Module):
    """Fuses frozen stride-16 foundation tokens with stride-4 RGB detail."""
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.GELU(),
                                nn.Conv2d(32,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU())
        self.mid=nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),nn.GroupNorm(8,128),nn.GELU())
        self.proj=nn.Conv2d(768,128,1)
        self.fuse8=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU())
        self.fuse4=nn.Sequential(nn.Conv2d(192,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                                 nn.Conv2d(64,6,1))

    def forward(self,features,size,raw):
        low=self.stem(raw)
        mid=self.mid(low)
        z=F.interpolate(self.proj(features.float()),size=mid.shape[-2:],
                        mode='bilinear',align_corners=False)
        z=self.fuse8(torch.cat((z,mid),1))
        z=F.interpolate(z,size=low.shape[-2:],mode='bilinear',align_corners=False)
        z=self.fuse4(torch.cat((z,low),1))
        return F.interpolate(z,size=size,mode='bilinear',align_corners=False)


def raw_input(x):
    mean=torch.as_tensor(MEAN.reshape(3),device=x.device)[None,:,None,None]
    std=torch.as_tensor(STD.reshape(3),device=x.device)[None,:,None,None]
    return (x*std+mean).clamp(0,1)


@torch.no_grad()
def features(backbone,x):
    return backbone.radio_model(x,feature_fmt='NCHW').features


def train(steps,batch,crop,fold,head_name,device):
    if device=='cuda':assert torch.cuda.is_available()
    ds=SatelliteCrops(samples=steps*batch,crop=crop,holdout_fold=fold)
    manifest={**ds.manifest,'model':'nvidia/C-RADIOv3-B','encoder_frozen':True,
              'steps':steps,'batch':batch,'head':head_name,'selection':'fixed final step'}
    (HERE/f'cradio_sp_f{fold}_{head_name}_manifest.json').write_text(json.dumps(manifest,indent=2))
    loader=DataLoader(ds,batch_size=batch,shuffle=False,
                      num_workers=1 if device=='cpu' else 3,pin_memory=device=='cuda')
    encoder=load_backbone(device)
    head=(DenseHeadHiRes() if head_name=='hires' else DenseHead()).to(device).train()
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01)
    class_weights=torch.tensor([.3,1.5,1.5,2.,2.,2.],device=device)
    for step,(x,y) in enumerate(loader,1):
        x=x.to(device,non_blocking=device=='cuda');y=y.to(device,non_blocking=device=='cuda')
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext()
        with amp:
            raw=raw_input(x)
            z=features(encoder,raw)
            logits=head(z,y.shape[-2:],raw) if head_name=='hires' else head(z,y.shape[-2:])
            loss=F.cross_entropy(logits.float(),y,weight=class_weights,ignore_index=255)
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        opt.step()
        if step%50==0 or step==1:print('step',step,'loss',round(float(loss),5),flush=True)
        if step%250==0 or step==steps:
            dest=HERE/f'cradio_sp_f{fold}_{head_name}_head_step{step}.pt'
            torch.save({k:v.detach().cpu().half() for k,v in head.state_dict().items()},dest)
            print('saved',dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--crop',type=int,default=512)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--head',choices=['basic','hires'],default='basic')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    train(a.steps,a.batch,a.crop,a.fold,a.head,a.device)
