"""Frozen EO-pretrained Swin-V2-B features for scene-aware Ae proposals."""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from torchvision.models import swin_v2_b

ROOT=Path('/home/nathan/Expedition Competition'); CW=ROOT/'Codex Work'
P=Path('/home/nathan/.cache/torch/hub/checkpoints/sentinel2_swinb_si_rgb-156a98d5.pth')

def prep(p):
    lo,hi=np.percentile(p,(2,98)); x=np.clip((p.astype(np.float32)-lo)/max(float(hi-lo),8),0,1)
    # Preserve film structure while reducing sheet-specific tone.
    x=cv2.resize(x,(224,224),interpolation=cv2.INTER_CUBIC)
    x=np.repeat(x[None],3,axis=0)
    return ((x-.5)/.25).astype(np.float32)

def main():
    d=pd.read_parquet(CW/'ae_sheet_selfadapt_features.parquet').reset_index(names='row_id')
    order=np.argsort(d.file.to_numpy()); out=np.empty((len(d),1024),np.float16)
    m=swin_v2_b(weights=None);m.load_state_dict(torch.load(P,map_location='cpu',weights_only=True));m.head=torch.nn.Identity()
    m=m.cuda().eval(); chipdir=ROOT/'data/ae_prep/chips'; batch=[]; ids=[]; cache_name=None;im=None
    def flush():
        nonlocal batch,ids
        if not batch:return
        x=torch.from_numpy(np.stack(batch)).cuda(non_blocking=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): z=m(x)
        out[np.asarray(ids)]=z.float().cpu().numpy().astype(np.float16);batch=[];ids=[]
    for n,j in enumerate(order,1):
        r=d.iloc[j]
        if r.file!=cache_name:
            cache_name=r.file;im=cv2.imread(str(chipdir/r.file),cv2.IMREAD_GRAYSCALE)
        p=im[int(r.py)-48:int(r.py)+48,int(r.px)-48:int(r.px)+48]
        if p.shape!=(96,96):continue
        batch.append(prep(p));ids.append(int(j))
        if len(batch)==16:flush()
        if n%1000==0:print(n,'/',len(d),flush=True)
    flush();np.save(CW/'ae_swin_eo_embeddings.npy',out)
    print('saved',out.shape,out.nbytes/1e6,'MB',flush=True)
if __name__=='__main__':main()
