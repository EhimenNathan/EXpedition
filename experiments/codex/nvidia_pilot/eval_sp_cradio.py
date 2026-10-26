"""Fixed-checkpoint C-RADIO fold-0 detection test with the official metric."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from rasterio.transform import Affine

ROOT=Path('/home/nathan/Expedition Competition')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(HERE)]
from koz3.hm import decode
from koz3.sp_fuse import fuse
from sp_assemble_oof import region_folds
from eval_sp_segformer import starts,evaluate,CLASSES
from train_sp_cradio import load_backbone,DenseHead,DenseHeadHiRes


@torch.no_grad()
def predict_chip(backbone,head,im,device,head_name):
    h,w=im.shape[:2]
    sums=np.zeros((5,h,w),np.float32)
    count=np.zeros((h,w),np.float32)
    for y0 in starts(h):
        for x0 in starts(w):
            crop=im[y0:y0+512,x0:x0+512]
            ch,cw=crop.shape[:2]
            padded=np.zeros((512,512,3),np.uint8);padded[:ch,:cw]=crop
            x=torch.from_numpy(padded.transpose(2,0,1).copy()[None]).to(device).float()/255
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext()
            with amp:
                z=backbone.radio_model(x,feature_fmt='NCHW').features
                logits=(head(z,(512,512),x) if head_name=='hires' else head(z,(512,512))).float()
            probs=logits.softmax(1)[0,1:,:ch,:cw].cpu().numpy()
            sums[:,y0:y0+ch,x0:x0+cw]+=probs
            count[y0:y0+ch,x0:x0+cw]+=1
    return sums/np.maximum(count,1e-6)


def export(checkpoint,device,fold,head_name):
    backbone=load_backbone(device)
    head=(DenseHeadHiRes() if head_name=='hires' else DenseHead()).to(device).eval()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    head.load_state_dict({k:v.float() for k,v in state.items()})
    rf=region_folds()
    ann=json.loads((ROOT/'data/spor_prep/ann.json').read_text())
    held=[c for c in ann if rf.get(c['region'])==fold]
    rows=[]
    for i,c in enumerate(held,1):
        im=cv2.imread(str(ROOT/'data/spor_prep/chips'/c['file']),cv2.IMREAD_COLOR)
        if im is None:raise FileNotFoundError(c['file'])
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        probs=predict_chip(backbone,head,im,device,head_name)
        valid=im.sum(2)>0
        transform=Affine(*c['transform'])
        tr=Transformer.from_crs(c['crs'],3857,always_xy=True)
        for cl,x,y,s in decode(probs,valid,thr=.15,nms=11):
            gx,gy=transform*(x,y);mx,my=tr.transform(gx,gy)
            rows.append((c['region'],CLASSES[cl],float(s),float(mx),float(my)))
        if i%20==0:print('inferred',i,'/',len(held),'raw objects',len(rows),flush=True)
    raw=pd.DataFrame(rows,columns=['region','cls','score','x','y'])
    raw.to_parquet(HERE/f'cradio_sp_f{fold}_{head_name}_raw.parquet')
    merged=fuse([raw],np.ones(1),match_m=4.) if len(raw) else raw
    merged.to_parquet(HERE/f'cradio_sp_f{fold}_{head_name}_merged.parquet')
    print('exported',len(raw),'raw',len(merged),'merged',flush=True)
    return merged


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path)
    p.add_argument('--score-only',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--head',choices=['basic','hires'],default='basic')
    a=p.parse_args()
    if a.score_only:pred=pd.read_parquet(HERE/f'cradio_sp_f{a.fold}_{a.head}_merged.parquet')
    else:
        if a.weights is None:p.error('--weights required')
        pred=export(a.weights,a.device,a.fold,a.head)
    evaluate(pred,'cradio_'+a.head,a.fold)
