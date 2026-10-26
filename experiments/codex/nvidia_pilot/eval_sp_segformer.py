"""Fold-0 NVIDIA SegFormer pilot inference and official-metric diagnostic.

Inference reads scene metadata and pixels only. Held-out labels enter only the
final scoring function. Threshold sweeps are diagnostic, not an honest choice
of a deploy threshold; use nested folds before selecting one.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyproj import Transformer
from rasterio.transform import Affine

ROOT = Path('/home/nathan/Expedition Competition')
HERE = ROOT / 'Codex Work/nvidia_pilot'
sys.path[:0] = [str(ROOT),str(ROOT/'scripts'),str(HERE)]
from koz3.hm import decode
from koz3.labels import to_gt_geojson
from koz3.score import score
from koz3.sp_fuse import fuse
from sp_assemble_oof import region_folds,models
from sp_fusion_oof import fc
from train_sp_segformer import MEAN,STD,model

CLASSES=('kurgany_povrezhdennye','kurgany_tselye','gorodishcha','arkhitektury','fortifikatsii')


def starts(length,size=512,stride=384):
    if length<=size:return [0]
    return sorted(set(range(0,length-size+1,stride))|{length-size})


@torch.no_grad()
def predict_chip(net,im):
    h,w=im.shape[:2]
    summ=np.zeros((5,h,w),np.float32)
    count=np.zeros((h,w),np.float32)
    for y0 in starts(h):
        for x0 in starts(w):
            crop=im[y0:y0+512,x0:x0+512]
            ch,cw=crop.shape[:2]
            padded=np.zeros((512,512,3),np.uint8);padded[:ch,:cw]=crop
            x=((padded.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1).copy()
            inp=torch.from_numpy(x[None]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=net(pixel_values=inp).logits.float()
            probs=F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False).softmax(1)
            p=probs[0,1:,:ch,:cw].cpu().numpy()
            summ[:,y0:y0+ch,x0:x0+cw]+=p
            count[y0:y0+ch,x0:x0+cw]+=1
    return summ/np.maximum(count,1e-6)


def export(weights, fold=0):
    net=model().cuda().eval()
    state=torch.load(weights,map_location='cpu',weights_only=True)
    net.load_state_dict({k:v.float() for k,v in state.items()})
    rf=region_folds()
    ann=json.loads((ROOT/'data/spor_prep/ann.json').read_text())
    held=[c for c in ann if rf.get(c['region'])==fold]
    assert held
    rows=[]
    for i,c in enumerate(held,1):
        im=cv2.imread(str(ROOT/'data/spor_prep/chips'/c['file']),cv2.IMREAD_COLOR)
        if im is None:raise FileNotFoundError(c['file'])
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        probs=predict_chip(net,im)
        valid=im.sum(2)>0
        T=Affine(*c['transform'])
        tr=Transformer.from_crs(c['crs'],3857,always_xy=True)
        for cl,x,y,s in decode(probs,valid,thr=.15,nms=11):
            gx,gy=T*(x,y);mx,my=tr.transform(gx,gy)
            rows.append((c['region'],CLASSES[cl],float(s),float(mx),float(my)))
        if i%20==0:print('inferred chips',i,'/',len(held),'raw objects',len(rows),flush=True)
    raw=pd.DataFrame(rows,columns=['region','cls','score','x','y'])
    raw.to_parquet(HERE/f'segformer_sp_f{fold}_raw.parquet')
    merged=fuse([raw],np.ones(1),match_m=4.) if len(raw) else raw
    merged.to_parquet(HERE/f'segformer_sp_f{fold}_merged.parquet')
    print('exported',len(raw),'raw',len(merged),'merged',flush=True)
    return merged


def evaluate(pred,candidate_name='segformer',fold=0):
    # Include held-out sites with zero detections, or recall is overstated.
    sites={s for s,f in region_folds().items() if f==fold}
    L=gpd.read_parquet(ROOT/'data/labels_3857.parquet')
    L=L[(L['mod']=='SpOr')&L.region.isin(sites)]
    gt=to_gt_geojson(L)
    print('fold',fold,'sites',len(sites),'GT',len(L),flush=True)
    ref=pd.read_parquet(ROOT/'work/spor/oof_hm_cands.parquet')
    ref=ref[ref.region.isin(sites)]
    members=models(extra=['dfnodn'])
    fused=fuse(list(members.values()),np.ones(len(members)))
    fused=fused[fused.score>=.25]
    gate=pd.read_parquet(ROOT/'work/spor/oof_scene_cls.parquet')
    fused=fused[~((fused.cls=='gorodishcha')&fused.region.isin(set(gate[gate.p<.1].region)))]
    added=gate[gate.p>=.1]
    fused=pd.concat([fused,pd.DataFrame({'region':added.region,'cls':'gorodishcha',
                                         'score':added.p,'x':added.x,'y':added.y3857})],ignore_index=True)
    fused=fused[fused.region.isin(sites)]
    print('v9_satellite_fusion threshold .25 weighted F1',round(score(fc(fused),gt)[0],5),
          'objects',len(fused),flush=True)
    for name,data in [('existing_heatmap',ref),(candidate_name,pred)]:
        for t in (.2,.25,.3,.4):
            p=data[data.score>=t]
            result,details=score(fc(p),gt)
            print(name,'threshold',t,'weighted F1',round(result,5),
                  'objects',len(p),'class F1',{k:round(details[k]['f1'],3) for k in CLASSES},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',type=Path)
    ap.add_argument('--score-only',action='store_true')
    ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args()
    if args.score_only:
        pred=pd.read_parquet(HERE/f'segformer_sp_f{args.fold}_merged.parquet')
    else:
        if args.weights is None:ap.error('--weights required unless --score-only')
        pred=export(args.weights,args.fold)
    evaluate(pred,fold=args.fold)
