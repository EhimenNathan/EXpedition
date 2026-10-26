"""Label-free whole-scene satellite appearance and quality descriptors.

Scene metadata's n_labels is deliberately excluded. Features use only the
image and sensor metadata available at inference, aggregated by region.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def image_features(path):
    a=np.load(path,mmap_mode='r')
    h,w=a.shape[1:]
    sy=max(1,int(np.ceil(h/512)));sx=max(1,int(np.ceil(w/512)))
    x=np.asarray(a[:3,::sy,::sx],np.float32)
    if x.max()>255:x=x/float(np.percentile(x,99.9)+1e-6)*255
    x=np.clip(x,0,255)
    gray=.299*x[0]+.587*x[1]+.114*x[2]
    valid=np.asarray(a[3,::sy,::sx]>0) if a.shape[0]>3 else np.ones(gray.shape,bool)
    if valid.mean()<.05:return None
    v=gray[valid]
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    e=np.hypot(gx,gy)[valid]
    gxx=np.mean(gx[valid]**2);gyy=np.mean(gy[valid]**2);gxy=np.mean(gx[valid]*gy[valid])
    smooth=cv2.GaussianBlur(gray,(0,0),sigmaX=6)
    sat=(x.max(0)-x.min(0))/(x.max(0)+1)
    return dict(img_gray_mean=float(v.mean()),img_gray_std=float(v.std()),
                img_gray_span=float(np.percentile(v,95)-np.percentile(v,5)),
                img_rgb_r=float(x[0][valid].mean()),img_rgb_g=float(x[1][valid].mean()),
                img_rgb_b=float(x[2][valid].mean()),img_sat=float(sat[valid].mean()),
                img_edge=float(np.mean(e)),img_edge_p90=float(np.percentile(e,90)),
                img_aniso=float(np.hypot(gxx-gyy,2*gxy)/(gxx+gyy+1e-6)),
                img_coarse_std=float(smooth[valid].std()),img_valid=float(valid.mean()))

def main():
    SC=pd.read_csv(ROOT/'kaggle_import/code/work/sp/scenes.csv')
    rows=[]
    for i,s in enumerate(SC.itertuples()):
        p=ROOT/'kaggle_import/code/work/sp/img'/f'{s.scene}.npy'
        d=image_features(p)
        if d is None:continue
        d.update(site=s.site,scene=s.scene,log_area=float(np.log(max(s.H*s.W,1))),
                 log_gsd=float(np.log(max(s.native_gsd,.01))),valid_meta=float(s.valid_frac))
        rows.append(d)
        if (i+1)%25==0:print(i+1,'scenes',flush=True)
    D=pd.DataFrame(rows)
    cols=[c for c in D if c not in ('site','scene')]
    G=D.groupby('site')[cols].median().reset_index().rename(columns={'site':'region'})
    G['log_scene_count']=np.log1p(D.groupby('site').size().reindex(G.region).to_numpy())
    G.to_parquet(HERE/'sp_scene_context_features.parquet',index=False)
    print('saved',len(D),'scenes',len(G),'regions',flush=True)

if __name__=='__main__':main()
