"""Build local candidate crops for a prospective geographic fold.

Heldout labels are included only as a separate evaluation array.  They are
never read by the trainer or used for checkpoint/threshold selection.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine

HERE=Path(__file__).resolve().parent
CW=HERE.parent
ROOT=CW.parent
S=96; HALF=S//2
EXCLUDE={0:set(),1:{'162_Курганы_33'},2:{'073_Тоссор','081_Костомаровское','129_Постройки_6','163_Курганы_34'},3:{'080_Верхний_Карабут','083_Барскоон'}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);a=ap.parse_args()
    F=pd.read_parquet(CW/'sp_morph_features.parquet').reset_index(names='row_id')
    SC=pd.read_csv(ROOT/'kaggle_import/code/work/sp/scenes.csv')
    train=[];train_y=[];train_site=[];train_row=[];test=[];test_y=[];test_row=[]
    for n,(region,q0) in enumerate(F.groupby('region')):
        if region in EXCLUDE[a.fold]:continue
        q=q0.copy();q['scene']=None;q['px']=-1;q['py']=-1;margin=np.full(len(q),-1)
        for sc in SC[SC.site==region].itertuples():
            T=Affine(*json.loads(sc.T));tr=Transformer.from_crs(3857,sc.crs,always_xy=True)
            X,Y=tr.transform(q.x.to_numpy(),q.y.to_numpy());ix,iy=(~T)*(X,Y)
            ix=np.asarray(ix).round().astype(int);iy=np.asarray(iy).round().astype(int)
            m=np.minimum.reduce([ix,iy,sc.W-1-ix,sc.H-1-iy]);use=(m>=HALF)&(m>margin)
            q.iloc[np.where(use)[0],q.columns.get_loc('scene')]=sc.scene
            q.iloc[np.where(use)[0],q.columns.get_loc('px')]=ix[use];q.iloc[np.where(use)[0],q.columns.get_loc('py')]=iy[use];margin[use]=m[use]
        for scene,g in q.groupby('scene',dropna=True):
            arr=np.load(ROOT/'kaggle_import/code/work/sp/img'/f'{scene}.npy',mmap_mode='r')
            for p in g.itertuples():
                im=np.asarray(arr[:3,p.py-HALF:p.py+HALF,p.px-HALF:p.px+HALF],np.uint8)
                if im.shape!=(3,S,S):continue
                if p.fold==a.fold:test.append(im);test_y.append(p.y_obj);test_row.append(p.row_id)
                else:train.append(im);train_y.append(p.y_obj);train_site.append(p.region);train_row.append(p.row_id)
        if (n+1)%25==0:print(n+1,len(train),len(test),flush=True)
    codes,names=pd.factorize(train_site);out=HERE/f'local_fold{a.fold}.npz'
    np.savez(out,train_img=np.stack(train),train_y=np.asarray(train_y,np.uint8),train_site=codes.astype(np.int16),
             train_row=np.asarray(train_row,np.int32),test_img=np.stack(test),test_y=np.asarray(test_y,np.uint8),test_row=np.asarray(test_row,np.int32))
    print(out,'train',len(train),'pos',sum(train_y),'sites',len(names),'test',len(test),'pos',sum(test_y),flush=True)

if __name__=='__main__':main()
