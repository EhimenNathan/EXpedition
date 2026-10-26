#!/usr/bin/env python3
import argparse,json,os,sys,zlib
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from pyproj import Transformer
from rasterio.transform import Affine
from ultralytics import YOLO
ROOT=Path('/home/nathan/Expedition Competition');HERE=ROOT/'Codex Work/ae_yolo_p2';SRC=ROOT/'data/ae_prep'
CLS=['kurgany_povrezhdennye','kurgany_tselye','gorodishcha','arkhitektury','fortifikatsii']
sys.path.insert(0,str(ROOT));from koz3.sp_fuse import fuse
def normalize(im):
 g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);v=g>0
 if v.sum()<100:return im
 p2,p50,p98=np.percentile(g[v],(2,50,98));gain=min(2.5,220/max(p98-p2,1.))
 x=np.clip((g.astype(np.float32)-p50)*gain+128,0,255).astype(np.uint8);x[~v]=0
 return cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)
def prepare(fold,out):
 rows=json.loads((SRC/'ann.json').read_text());tr=[r for r in rows if r['fold']!=fold and (r['boxes'] or zlib.crc32(r['file'].encode())/2**32<.35)];held=[r for r in rows if r['fold']==fold]
 I=out/'train/images';L=out/'train/labels';I.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
 for i,r in enumerate(tr):
  stem=f'{i:05d}';dest=I/f'{stem}.jpg'
  if not dest.exists():
   im=cv2.imread(str(SRC/'chips'/r['file']));cv2.imwrite(str(dest),normalize(im),[cv2.IMWRITE_JPEG_QUALITY,95])
  with open(L/f'{stem}.txt','w') as f:
   for (x0,y0,x1,y1),c in zip(r['boxes'],r['classes']):
    x0,x1=sorted((max(0,min(r['w'],x0)),max(0,min(r['w'],x1))));y0,y1=sorted((max(0,min(r['h'],y0)),max(0,min(r['h'],y1))))
    if x1-x0>=2 and y1-y0>=2:f.write(f"{c} {(x0+x1)/(2*r['w']):.7f} {(y0+y1)/(2*r['h']):.7f} {(x1-x0)/r['w']:.7f} {(y1-y0)/r['h']:.7f}\n")
 y=out/'data.yaml';y.write_text('path: '+str(out/'train')+'\ntrain: images\nval: images\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(CLS)))
 (out/'manifest.json').write_text(json.dumps({'fold':fold,'train':len(tr),'held':len(held),'train_sites':sorted({r['region'] for r in tr}),'held_sites':sorted({r['region'] for r in held})},ensure_ascii=False,indent=2));print('prepared',len(tr),len(held),flush=True);return y,held
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--epochs',type=int,default=50);a=ap.parse_args();out=HERE/f'fold{a.fold}';y,held=prepare(a.fold,out)
 model=YOLO(str(HERE/'yolo11m-p2.yaml')).load(str(ROOT/'Codex Work/yolo11m.pt'))
 model.train(data=str(y),epochs=a.epochs,imgsz=1024,batch=2,device=0,workers=4,project=str(out),name='run',exist_ok=True,seed=20261026+a.fold,hsv_h=0,hsv_s=0,hsv_v=.15,degrees=180,fliplr=.5,flipud=.5,mosaic=1,close_mosaic=10,amp=True,val=False,plots=False,save=True,verbose=True,patience=a.epochs+1)
 w=out/'run/weights/last.pt';net=YOLO(str(w));rows=[]
 for i in range(0,len(held),2):
  meta=held[i:i+2];ims=[normalize(cv2.imread(str(SRC/'chips'/r['file']))) for r in meta];pred=net.predict(ims,imgsz=1024,conf=.03,iou=.5,max_det=1000,device=0,batch=2,verbose=False)
  for r,p in zip(meta,pred):
   T=Affine(*r['transform']);project=Transformer.from_crs(r['crs'],3857,always_xy=True);scene=r['file'].split('__')[1]
   for b,s,c in zip(p.boxes.xyxy.cpu().numpy(),p.boxes.conf.cpu().numpy(),p.boxes.cls.cpu().numpy().astype(int)):
    gx,gy=T*((b[0]+b[2])/2,(b[1]+b[3])/2);x,y=project.transform(gx,gy);rows.append((r['region'],scene,CLS[c],float(s),x,y,a.fold))
  if i%40==0:print('infer',i,len(rows),flush=True)
 d=pd.DataFrame(rows,columns=['region','scene','cls','score','x','y','fold']);d.to_parquet(out/'oof_raw.parquet')
 parts=[]
 for (reg,scene),g in d.groupby(['region','scene']):
  q=fuse([g[['region','cls','score','x','y']]],[1.],match_m=4.);q['scene']=scene;q['fold']=a.fold;parts.append(q)
 pd.concat(parts,ignore_index=True).to_parquet(out/'oof_layers.parquet');print('done',len(d),sum(map(len,parts)),flush=True)
if __name__=='__main__':main()
