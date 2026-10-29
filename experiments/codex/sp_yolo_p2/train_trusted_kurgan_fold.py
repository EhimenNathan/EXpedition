#!/usr/bin/env python3
"""P2 kurgan proposal model with partial-label-aware supervision.

Satellite regions are mostly annotated for one archaeological family.  A
normal multi-class detector therefore learns thousands of false negatives from
architecture/hillfort-only regions.  This binary model trains only on regions
where kurgans were actually annotated; all kurgan boxes map to one proposal
class.  Held-out inference still covers every scene in the geographic fold.
"""
import argparse,json,os
from pathlib import Path
import cv2,numpy as np,pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from ultralytics import YOLO

ROOT=Path('/home/nathan/Expedition Competition/Codex Work/dinov3_wt')
SRC=ROOT/'data/spor_prep'; HERE=ROOT/'experiments/codex/sp_yolo_p2'

def prepare(fold,out):
    rows=json.loads((SRC/'ann.json').read_text())
    fp=ROOT/'kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/folds.json'
    ext={k.rsplit('__',1)[0]:int(v) for k,v in json.loads(fp.read_text()).items()}
    for r in rows:
        if r['region'] in ext:r['fold']=ext[r['region']]
    trusted={r['region'] for r in rows if any(c in (0,1) for c in r['classes'])}
    train=[r for r in rows if int(r['fold'])!=fold and r['region'] in trusted]
    held=[r for r in rows if int(r['fold'])==fold]
    imd,lad=out/'dataset/images',out/'dataset/labels';imd.mkdir(parents=True,exist_ok=True);lad.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(train):
        stem=f'{i:05d}';dst=imd/f'{stem}.jpg'
        if not dst.exists():os.link(SRC/'chips'/r['file'],dst)
        with (lad/f'{stem}.txt').open('w') as f:
            for (x0,y0,x1,y1),c in zip(r['boxes'],r['classes']):
                if c not in (0,1):continue
                x0,x1=sorted((np.clip(x0,0,r['w']),np.clip(x1,0,r['w'])));y0,y1=sorted((np.clip(y0,0,r['h']),np.clip(y1,0,r['h'])))
                if x1-x0>=2 and y1-y0>=2:
                    f.write(f"0 {(x0+x1)/(2*r['w']):.8f} {(y0+y1)/(2*r['h']):.8f} {(x1-x0)/r['w']:.8f} {(y1-y0)/r['h']:.8f}\n")
    y=out/'data.yaml';y.write_text(f'path: {out/"dataset"}\ntrain: images\nval: images\nnames:\n  0: kurgan_any\n')
    (out/'manifest.json').write_text(json.dumps({'fold':fold,'trusted_regions':sorted(trusted),'train_regions':sorted({r['region'] for r in train}),'train_chips':len(train),'heldout_chips':len(held)},ensure_ascii=False,indent=2))
    return y,held

def export(net,held,fold,out):
    z=[]
    for start in range(0,len(held),2):
        meta=held[start:start+2];images=[cv2.imread(str(SRC/'chips'/r['file'])) for r in meta]
        pp=net.predict(images,imgsz=1024,conf=.005,iou=.5,max_det=1500,device=0,batch=2,verbose=False)
        for r,p in zip(meta,pp):
            t=Affine(*r['transform']);pr=Transformer.from_crs(r['crs'],3857,always_xy=True);scene=r['file'].rsplit('__',1)[0]
            for b,s in zip(p.boxes.xyxy.cpu().numpy(),p.boxes.conf.cpu().numpy()):
                gx,gy=t*((b[0]+b[2])/2,(b[1]+b[3])/2);x,y=pr.transform(gx,gy)
                z.append((r['region'],scene,fold,'kurgany_povrezhdennye',float(s),x,y))
        if start%20==0:print('infer',start,'/',len(held),'candidates',len(z),flush=True)
    pd.DataFrame(z,columns=['region','scene','fold','cls','score','x','y']).to_parquet(out/'oof_candidates.parquet')
    print('saved',len(z),'candidates',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--fold',type=int,required=True);a.add_argument('--epochs',type=int,default=40);q=a.parse_args()
    out=HERE/f'fold{q.fold}_trustedk';out.mkdir(parents=True,exist_ok=True);yaml,held=prepare(q.fold,out)
    m=YOLO(str(HERE/'yolo11m-p2.yaml')).load(str(ROOT.parent/'yolo11m.pt'))
    m.train(data=str(yaml),epochs=q.epochs,imgsz=1024,batch=2,device=0,workers=4,project=str(out),name='run',exist_ok=True,
            seed=20261030+q.fold,hsv_h=.015,hsv_s=.35,hsv_v=.25,degrees=180,translate=.1,scale=.35,fliplr=.5,flipud=.5,
            mosaic=1.,close_mosaic=8,mixup=.1,amp=True,val=False,plots=False,save=True,verbose=True,patience=q.epochs+1)
    export(YOLO(str(out/'run/weights/last.pt')),held,q.fold,out)

if __name__=='__main__':main()
