"""Independent NVIDIA MiT-B1 satellite segmentation pilot, geographic fold 0.

The encoder checkpoint is licensed for non-commercial research/evaluation.
Training here is an experiment only; do not put its weights in a competition
submission until the applicable license has been reviewed for that use.
No held-out site contributes pixels, labels, augmentation statistics, or model
selection. A fixed final-step checkpoint is saved under Codex Work.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rasterio.features import rasterize
from shapely.affinity import translate
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

ROOT = Path('/home/nathan/Expedition Competition')
HERE = ROOT / 'Codex Work/nvidia_pilot'
SOURCE = ROOT / 'data/spor_prep/chips'
sys.path.insert(0, str(ROOT))
from koz3.sp_train import load_chips

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


class SatelliteCrops(Dataset):
    def __init__(self, samples=12000, crop=512, holdout_fold=0):
        all_chips = load_chips('SpOr')
        self.chips = [c for c in all_chips if c['fold'] != holdout_fold]
        train_sites = {c['region'] for c in self.chips}
        held_sites = {c['region'] for c in all_chips if c['fold'] == holdout_fold}
        assert train_sites.isdisjoint(held_sites)
        self.sites = sorted(train_sites)
        self.by_site = {s:[c for c in self.chips if c['region'] == s] for s in self.sites}
        self.samples, self.crop = samples, crop
        self.cache = {}
        self.manifest = {'fold':holdout_fold, 'train_sites':self.sites,
                         'heldout_sites':sorted(held_sites),
                         'train_chips':len(self.chips), 'crop':crop,
                         'site_sampling':'uniform', 'heldout_used_for_training':False}

    def __len__(self):
        return self.samples

    def _image(self, filename):
        im = self.cache.get(filename)
        if im is None:
            im = cv2.imread(str(SOURCE / filename), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(filename)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if len(self.cache) < 50:
                self.cache[filename] = im
        return im

    def __getitem__(self, index):
        rng = np.random.default_rng(20261024 + index)
        site = self.sites[int(rng.integers(len(self.sites)))]
        chips = self.by_site[site]
        c = chips[int(rng.integers(len(chips)))]
        im = self._image(c['file'])
        h, w = im.shape[:2]
        side = self.crop
        if c['objs'] and rng.random() < .7:
            _, g = c['objs'][int(rng.integers(len(c['objs'])))]
            p = g.representative_point()
            cx, cy = p.x + rng.normal(0, 60), p.y + rng.normal(0, 60)
        else:
            cx, cy = rng.uniform(0,w), rng.uniform(0,h)
        x0 = int(np.clip(cx-side/2, 0, max(0,w-side)))
        y0 = int(np.clip(cy-side/2, 0, max(0,h-side)))
        crop = im[y0:y0+side,x0:x0+side].copy()
        if crop.shape[:2] != (side,side):
            crop = cv2.copyMakeBorder(crop,0,side-crop.shape[0],0,side-crop.shape[1],
                                      cv2.BORDER_CONSTANT,value=0)
        shapes = [(translate(g,xoff=-x0,yoff=-y0),int(cl)+1)
                  for cl,g in c['objs'] if not g.is_empty]
        mask = rasterize(shapes,out_shape=(side,side),fill=0,dtype=np.uint8) if shapes else np.zeros((side,side),np.uint8)
        valid = crop.sum(2)>0
        mask[~valid]=255
        # Preserve a substantial raw-image path for clear sources.
        if rng.random() < .3:
            gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
            pix = gray[valid]
            if len(pix)>100:
                p2,p50,p98 = np.percentile(pix,[2,50,98])
                if p98-p2<80:
                    gain=min(2.5,220/max(1,p98-p2))
                    crop=np.clip((crop.astype(np.float32)-p50)*gain+128,0,255).astype(np.uint8)
                    crop[~valid]=0
        if rng.random()<.25:
            crop=np.clip(crop.astype(np.float32)*rng.uniform(.85,1.15)+rng.uniform(-15,15),0,255).astype(np.uint8)
            crop[~valid]=0
        if rng.random()<.1:
            crop=cv2.GaussianBlur(crop,(3,3),rng.uniform(.3,1.2))
        if rng.random()<.1:
            q=int(rng.integers(60,91))
            ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,q])
            if ok:
                crop=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        k=int(rng.integers(4));crop=np.rot90(crop,k).copy();mask=np.rot90(mask,k).copy()
        if rng.random()<.5:
            crop=crop[:,::-1].copy();mask=mask[:,::-1].copy()
        x=(crop.astype(np.float32)/255-MEAN)/STD
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(mask.astype(np.int64))


def model():
    os.environ.setdefault('HF_HOME',str(HERE/'hf_cache'))
    return SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b1',num_labels=6,ignore_mismatched_sizes=True,
        id2label={i:n for i,n in enumerate(('background','damaged_kurgan','intact_kurgan',
                                            'hillfort','architecture','fortification'))})


def train(steps,batch,fold=0):
    assert torch.cuda.is_available(), 'Wait until the GPU is free.'
    ds=SatelliteCrops(samples=steps*batch,holdout_fold=fold)
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/f'segformer_sp_f{fold}_manifest.json').write_text(json.dumps(ds.manifest,ensure_ascii=False,indent=2))
    loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=3,pin_memory=True)
    net=model().cuda().train()
    opt=torch.optim.AdamW([{'params':net.segformer.parameters(),'lr':5e-5},
                           {'params':net.decode_head.parameters(),'lr':2e-4}],weight_decay=.01)
    weights=torch.tensor([.25,1.5,1.5,2.,2.,2.],device='cuda')
    for step,(x,y) in enumerate(loader,1):
        x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=F.interpolate(net(pixel_values=x).logits.float(),size=y.shape[-2:],
                                 mode='bilinear',align_corners=False)
            loss=F.cross_entropy(logits,y,weight=weights,ignore_index=255)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        if step%100==0 or step==1:
            print('step',step,'loss',round(loss.item(),5),flush=True)
        if step%500==0 or step==steps:
            path=HERE/f'segformer_sp_f{fold}_step{step}.pt'
            torch.save({k:v.detach().cpu().half() for k,v in net.state_dict().items()},path)
            print('saved',path,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--batch',type=int,default=2)
    ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args()
    if args.smoke:
        ds=SatelliteCrops(samples=4,crop=128)
        x,y=ds[0]
        assert x.shape==(3,128,128) and y.shape==(128,128)
        assert set(torch.unique(y).tolist()).issubset({0,1,2,3,4,5,255})
        print('smoke',x.shape,torch.unique(y,return_counts=True))
    else:
        train(args.steps,args.batch,args.fold)
