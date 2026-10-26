"""Visibility-curriculum aerial point detector.

The Ae labels mix visually observable cemetery film with archaeologically valid
objects that have almost no film contrast.  Training every polygon as an image
positive made earlier dense models fire on generic grey texture.  This pilot
selects *training* sites using predictions made out-of-fold by the existing
FCOS model, then learns all kurgan points on the observable sites while using
the remaining training sites only as background.  The target fold never enters
site selection or optimization.

Selection is fixed: official-metric precision >= .20 and at least 8 kurgan TP
from the FCOS OOF member at the deployed-style .18 output threshold.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

ROOT = Path('/home/nathan/Expedition Competition')
CW = ROOT / 'Codex Work'

import sys
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(CW/'ae_yolo_pilot'), str(CW)]
from ae_dense_p2 import P2Net, MEAN, STD
from ae_dense_heatmap import DenseSampler, CROP, KCLS
from koz3 import hm
from koz3.labels import to_gt_geojson
from koz3.sp_train import CLS, load_chips, prep_dir
from ae_scale_fusion_probe import model_input, make, evaluate


def select_sites(target_fold):
    ann = json.loads((ROOT/'data/ae_prep/ann.json').read_text())
    train_sites = {x['region'] for x in ann if x['fold'] != target_fold}
    labels = geopandas_labels = gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    labels = labels[(labels['mod'] == 'Ae') & labels.region.isin(train_sites)]
    raw = model_input(ROOT/'work/ae/oof_ae_fcos_fast_cands.parquet', train_sites, .2)
    pred = make([raw])
    chosen, audit = set(), []
    for site in sorted(train_sites):
        ev = evaluate(pred[pred.region == site], to_gt_geojson(labels[labels.region == site]))
        cc = ev['classes']; ks = ('kurgany_povrezhdennye', 'kurgany_tselye')
        tp = sum(cc[c]['tp'] for c in ks); fp = sum(cc[c]['fp'] for c in ks)
        pm = tp / max(tp + .5*fp, 1)
        keep = tp >= 8 and pm >= .20
        if keep: chosen.add(site)
        audit.append(dict(site=site, tp=tp, fp=fp, precision_metric=pm, selected=keep))
    return chosen, audit


class VisibleSampler(DenseSampler):
    def __init__(self, chips, selected, seed):
        super().__init__(chips, seed)
        self.selected = set(selected)
        self.all_by_site, self.pos_by_site = {}, {}
        for i, c in enumerate(chips):
            self.all_by_site.setdefault(c['region'], []).append(i)
            if c['region'] in self.selected and self.kobjs[i]:
                self.pos_by_site.setdefault(c['region'], []).append(i)
        self.all_sites = sorted(self.all_by_site)
        self.pos_sites = sorted(self.pos_by_site)
        assert self.pos_sites

    def crop(self):
        positive = self.rng.random() < .70
        if positive:
            site = self.pos_sites[int(self.rng.integers(len(self.pos_sites)))]
            ii = self.pos_by_site[site]
            i = ii[int(self.rng.integers(len(ii)))]
        else:
            site = self.all_sites[int(self.rng.integers(len(self.all_sites)))]
            ii = self.all_by_site[site]
            i = ii[int(self.rng.integers(len(ii)))]
        c = self.chips[i]; im = self.img(c); h, w = im.shape[:2]
        objs = self.kobjs[i] if site in self.selected else []
        if positive and objs:
            p = objs[int(self.rng.integers(len(objs)))][1].representative_point()
            cx, cy = p.x + self.rng.normal(0, 96), p.y + self.rng.normal(0, 96)
        else:
            cx, cy = self.rng.uniform(0, w), self.rng.uniform(0, h)
        x0 = float(np.clip(cx-CROP/2, min(0,w-CROP), max(0,w-CROP)))
        y0 = float(np.clip(cy-CROP/2, min(0,h-CROP), max(0,h-CROP)))
        M = np.array([[1,0,-x0],[0,1,-y0]], np.float32)
        out = cv2.warpAffine(np.ascontiguousarray(im), M, (CROP,CROP), borderValue=0)
        from shapely.affinity import affine_transform
        oo = [(KCLS[cl], affine_transform(g,[1,0,0,1,-x0,-y0])) for cl,g in objs]
        ch, _ = hm.draw_targets((CROP,CROP), oo, 2, sig_frac=0, sig_min=1.5, sig_max=1.5)
        target = np.concatenate([np.maximum(ch[0],ch[1])[None], ch])
        # Film invariance, without inventing spatial objects.
        g = out.astype(np.float32).mean(2)
        g = g*self.rng.uniform(.72,1.32) + self.rng.uniform(-28,28)
        if self.rng.random() < .45: g = 255*np.clip(g/255,0,1)**self.rng.uniform(.65,1.5)
        if self.rng.random() < .25: g = cv2.GaussianBlur(g,(int(self.rng.choice([3,5])),)*2,0)
        if self.rng.random() < .30: g += self.rng.normal(0,self.rng.uniform(2,9),g.shape)
        val = (out.max(2)>0).astype(np.float32)
        img = np.repeat(np.clip(g,0,255)[None],3,0); img=(img-MEAN)/STD*val
        k=int(self.rng.integers(4)); fl=self.rng.random()<.5
        img=np.rot90(img,k,axes=(-2,-1)); target=np.rot90(target,k,axes=(-2,-1)); val=np.rot90(val,k)
        if fl: img=img[...,::-1]; target=target[...,::-1]; val=val[...,::-1]
        return [np.ascontiguousarray(x,np.float32) for x in (img,target,val)]


def train(fold, iters, bs):
    selected, audit = select_sites(fold)
    chips=[c for c in load_chips('Ae') if c['fold'] != fold]
    s=VisibleSampler(chips,selected,20261026+fold); torch.manual_seed(20261026+fold)
    print('target_fold',fold,'selected',sorted(selected),flush=True)
    m=P2Net().cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,3e-4,iters,pct_start=.05)
    ex=ThreadPoolExecutor(6); fut=[ex.submit(s.batch,bs) for _ in range(3)]; t0=time.time()
    for it in range(iters):
        img,t,v=[x.cuda(non_blocking=True) for x in fut.pop(0).result()]; fut.append(ex.submit(s.batch,bs))
        with torch.autocast('cuda',dtype=torch.bfloat16): o=m(img.to(memory_format=torch.channels_last))
        w=v[:,None]; loss=hm.focal_hm(o[:,:1],t[:,:1],w)+.4*hm.focal_hm(o[:,1:],t[:,1:],w)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step(); sch.step()
        if it%250==0 or it==iters-1: print('visible fold',fold,'it',it,'loss',float(loss),'sec',round(time.time()-t0),flush=True)
    path=CW/f'ae_visible_p2_f{fold}.pt'; torch.save({k:v.half() for k,v in m.state_dict().items()},path)
    (CW/f'ae_visible_p2_f{fold}_selection.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))


@torch.no_grad()
def infer(fold, thr=.02):
    from ae_dense_p2 import infer as old_infer
    # old_infer has a fixed filename; temporarily duplicate the small state file.
    src=CW/f'ae_visible_p2_f{fold}.pt'; dst=CW/f'ae_dense_p2_f{fold}.pt'
    backup=CW/f'ae_dense_p2_f{fold}.pt.visibility_backup'
    if dst.exists(): dst.rename(backup)
    try:
        import shutil; shutil.copy2(src,dst)
        old_infer(fold,thr=thr,tag='visible')
    finally:
        dst.unlink(missing_ok=True)
        if backup.exists(): backup.rename(dst)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True)
    ap.add_argument('--iters',type=int,default=5000); ap.add_argument('--bs',type=int,default=4)
    ap.add_argument('--train',action='store_true'); ap.add_argument('--infer',action='store_true')
    a=ap.parse_args()
    if a.train: train(a.fold,a.iters,a.bs)
    if a.infer: infer(a.fold)
