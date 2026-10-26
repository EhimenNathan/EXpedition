"""Private Kaggle pilot data: satellite candidate crops, geographic fold 3.

Train labels are from folds 0-2 only. Fold-3 crops and row IDs carry no GT
labels or evaluation statistics into the uploaded archive.
"""
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
S=96
HALF=S//2
BUFFER_EXCLUDE={'080_Верхний_Карабут','083_Барскоон'}

def main():
    F=pd.read_parquet(CW/'sp_morph_features.parquet').reset_index(names='row_id')
    SC=pd.read_csv(ROOT/'kaggle_import/code/work/sp/scenes.csv')
    train=[];train_y=[];train_site=[];train_row=[]
    test=[];test_row=[]
    for n,(region,q) in enumerate(F.groupby('region')):
        if region in BUFFER_EXCLUDE:
            continue
        q=q.copy();q['scene']=None;q['px']=-1;q['py']=-1
        margin=np.full(len(q),-1)
        for sc in SC[SC.site==region].itertuples():
            T=Affine(*json.loads(sc.T));tr=Transformer.from_crs(3857,sc.crs,always_xy=True)
            X,Y=tr.transform(q.x.to_numpy(),q.y.to_numpy())
            ix,iy=(~T)*(X,Y)
            ix=np.asarray(ix).round().astype(int);iy=np.asarray(iy).round().astype(int)
            m=np.minimum.reduce([ix,iy,sc.W-1-ix,sc.H-1-iy])
            use=(m>=HALF)&(m>margin)
            q.iloc[np.where(use)[0],q.columns.get_loc('scene')]=sc.scene
            q.iloc[np.where(use)[0],q.columns.get_loc('px')]=ix[use]
            q.iloc[np.where(use)[0],q.columns.get_loc('py')]=iy[use]
            margin[use]=m[use]
        for scene,g in q.groupby('scene',dropna=True):
            arr=np.load(ROOT/'kaggle_import/code/work/sp/img'/f'{scene}.npy',mmap_mode='r')
            for p in g.itertuples():
                im=np.asarray(arr[:3,p.py-HALF:p.py+HALF,p.px-HALF:p.px+HALF],np.uint8)
                if im.shape!=(3,S,S):continue
                if p.fold==3:
                    test.append(im);test_row.append(p.row_id)
                else:
                    train.append(im);train_y.append(p.y_obj);train_site.append(p.region);train_row.append(p.row_id)
        if (n+1)%20==0:print(n+1,'regions',len(train),len(test),flush=True)
    stage=HERE/'upload';stage.mkdir(exist_ok=True)
    site_codes,site_names=pd.factorize(train_site)
    np.savez_compressed(stage/'candidate_crops_f3.npz',
        train_img=np.stack(train),train_y=np.asarray(train_y,np.uint8),
        train_site=site_codes.astype(np.int16),train_row=np.asarray(train_row,np.int32),
        test_img=np.stack(test),test_row=np.asarray(test_row,np.int32))
    z=np.load(stage/'candidate_crops_f3.npz')
    assert 'test_y' not in z.files and 'test_label' not in z.files
    assert set(F.loc[z['train_row'],'fold'])<=set((0,1,2))
    assert set(F.loc[z['test_row'],'fold'])=={3}
    print('train',len(train),'positive',sum(train_y),'sites',len(site_names),'test',len(test),
          'test sites',F.loc[test_row,'region'].nunique(),'npz_mb',round((stage/'candidate_crops_f3.npz').stat().st_size/1e6,1),flush=True)

if __name__=='__main__':main()
