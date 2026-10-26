"""Small, fixed C-RADIOv3-B satellite feature probe on geographic fold 0.

This is a diagnostic of frozen features, not an official object-detection score.
Held-out sites are used only once, for final reporting; the classifier and
decision threshold are selected from training-site features alone.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from rasterio.features import rasterize
from safetensors.torch import load_file
from shapely.affinity import translate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from transformers import AutoConfig, AutoModel

ROOT = Path('/home/nathan/Expedition Competition')
HERE = ROOT / 'Codex Work/nvidia_pilot'
sys.path.insert(0, str(ROOT))
from koz3.sp_train import load_chips


def crop_sample(c, rng, side=128):
    im = cv2.imread(str(ROOT/'data/spor_prep/chips'/c['file']), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(c['file'])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    if c['objs']:
        _, g = c['objs'][int(rng.integers(len(c['objs'])))]
        p = g.representative_point()
        cx, cy = p.x, p.y
    else:
        cx, cy = rng.uniform(0,w), rng.uniform(0,h)
    x0 = int(np.clip(cx-side/2,0,max(0,w-side)))
    y0 = int(np.clip(cy-side/2,0,max(0,h-side)))
    crop = im[y0:y0+side,x0:x0+side].copy()
    if crop.shape[:2] != (side,side):
        crop = cv2.copyMakeBorder(crop,0,side-crop.shape[0],0,side-crop.shape[1],
                                  cv2.BORDER_CONSTANT,value=0)
    shapes = [(translate(g,xoff=-x0,yoff=-y0),1) for _,g in c['objs'] if not g.is_empty]
    mask = rasterize(shapes,out_shape=(side,side),fill=0,dtype=np.uint8) if shapes else np.zeros((side,side),np.uint8)
    valid = crop.sum(2)>0
    return crop,mask,valid


@torch.inference_mode()
def extract(model,c,rng):
    image,mask,valid=crop_sample(c,rng)
    x=torch.from_numpy(image.transpose(2,0,1).copy()[None]).float()/255
    output=model(x)
    tokens=output.features[0].float().cpu().numpy()
    n=tokens.shape[0]
    grid=int(np.sqrt(n))
    assert grid*grid==n
    labels=cv2.resize(mask,(grid,grid),interpolation=cv2.INTER_NEAREST).reshape(-1)
    keep=cv2.resize(valid.astype(np.uint8),(grid,grid),interpolation=cv2.INTER_NEAREST).reshape(-1)>0
    rgb=cv2.resize(image,(grid,grid),interpolation=cv2.INTER_AREA).reshape(-1,3).astype(np.float32)/255
    return tokens[keep],rgb[keep],labels[keep]


def main():
    path=HERE/'cradio_v3b'
    config=AutoConfig.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    model=AutoModel.from_config(config,trust_remote_code=True).eval()
    model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
    chips=load_chips('SpOr')
    train=[c for c in chips if c['fold']!=0 and c['objs']]
    held=[c for c in chips if c['fold']==0 and c['objs']]
    train_sites={c['region'] for c in train}
    held_sites={c['region'] for c in held}
    assert train_sites.isdisjoint(held_sites)
    rng=np.random.default_rng(20261024)
    # One positive-centred chip per training site. This probes feature transfer,
    # while limiting CPU and preventing a prolific site dominating the fit.
    tr=[]
    for site in sorted(train_sites):
        options=[c for c in train if c['region']==site]
        tr.append(options[int(rng.integers(len(options)))])
    va=[]
    for site in sorted(held_sites):
        options=[c for c in held if c['region']==site]
        va.append(options[int(rng.integers(len(options)))])
    xtr,rtr,ytr=[],[],[]
    for i,c in enumerate(tr,1):
        x,r,y=extract(model,c,rng);xtr.append(x);rtr.append(r);ytr.append(y)
        if i%20==0:print('train sites',i,'/',len(tr),flush=True)
    xtr=np.concatenate(xtr);rtr=np.concatenate(rtr);ytr=np.concatenate(ytr)
    xva,rva,yva=[],[],[]
    for i,c in enumerate(va,1):
        x,r,y=extract(model,c,rng);xva.append(x);rva.append(r);yva.append(y)
        if i%10==0:print('heldout sites',i,'/',len(va),flush=True)
    xva=np.concatenate(xva);rva=np.concatenate(rva);yva=np.concatenate(yva)
    assert ytr.sum()>0 and yva.sum()>0
    clf=LogisticRegression(max_iter=400,class_weight='balanced',C=.1)
    clf.fit(xtr,ytr)
    ptr=clf.predict_proba(xtr)[:,1]
    pva=clf.predict_proba(xva)[:,1]
    rgb_clf=LogisticRegression(max_iter=400,class_weight='balanced',C=.1)
    rgb_clf.fit(rtr,ytr)
    rgb_pva=rgb_clf.predict_proba(rva)[:,1]
    thresholds=np.arange(.05,.96,.05)
    train_f1=[f1_score(ytr,ptr>=t,zero_division=0) for t in thresholds]
    t=float(thresholds[int(np.argmax(train_f1))])
    result={'model':'nvidia/C-RADIOv3-B frozen linear probe',
            'geographic_fold':0,'train_sites':len(tr),'heldout_sites':len(va),
            'train_tokens':len(ytr),'heldout_tokens':len(yva),
            'train_positive_fraction':float(ytr.mean()),
            'heldout_positive_fraction':float(yva.mean()),
            'threshold_from_train':t,
            'heldout_roc_auc':float(roc_auc_score(yva,pva)),
            'heldout_average_precision':float(average_precision_score(yva,pva)),
            'heldout_token_f1':float(f1_score(yva,pva>=t,zero_division=0)),
            'raw_rgb_heldout_roc_auc':float(roc_auc_score(yva,rgb_pva)),
            'raw_rgb_heldout_average_precision':float(average_precision_score(yva,rgb_pva)),
            'limitation':'positive-centred 128px crops; token metric is not official detection F1'}
    (HERE/'cradio_linear_probe_f0.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
