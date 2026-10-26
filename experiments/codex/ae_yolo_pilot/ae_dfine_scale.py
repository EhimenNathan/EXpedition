"""Ae-only small-object D-FINE fine-tune at 1.5x pixel scale.

Reads Claude's completed joint fold checkpoint but writes only under Codex
Work. Run a single fold first when the local GPU is free. Source images and
boxes are scaled together; affine transforms are divided by the same factor,
so detections still map to the original ground coordinates.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ['DET_ARCH']='dfine'
os.environ['DET_JOINT']='0'
os.environ['DET_QAUG']='1'
os.environ['HF_HUB_OFFLINE']='1'

ROOT=Path(os.environ.get('KOZ3_ROOT','/home/nathan/Expedition Competition'))
OUT=ROOT/'Codex Work/ae_yolo_pilot/dfine_scale'
sys.path.insert(0,str(ROOT))
SCALE=1.5
TAG='codex_ae_s15'

def gpu_clear():
    if os.environ.get('KOZ3_ISOLATED_GPU')=='1':
        return
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader'],text=True)
    users=[]
    for line in output.splitlines():
        pid,mem=(x.strip() for x in line.split(',',1))
        if int(mem.split()[0])>=1000 and int(pid)!=os.getpid():
            users.append((pid,mem))
    if users:raise RuntimeError(f'GPU in use: {users}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,choices=range(4))
    ap.add_argument('--eval-fold',type=int,choices=range(4))
    ap.add_argument('--iters',type=int,default=1200)
    args=ap.parse_args()
    assert (args.fold is None)!=(args.eval_fold is None)
    gpu_clear()
    import cv2
    import torch
    from rasterio.transform import Affine
    from koz3 import ae_det_train as a
    config=ROOT/'hf_dfine_config'
    if config.is_dir():a.DFINE=str(config)
    a.ROOT=str(OUT)
    original_chips=a.chips
    original_image=a.image

    def scaled_chips():
        out=original_chips()
        for c in out:
            c['w']=round(c['w']*SCALE);c['h']=round(c['h']*SCALE)
            c['B'][:,:4]*=SCALE
            c['transform']=list(Affine(*c['transform'])*Affine.scale(1/SCALE))[:6]
        return out

    def scaled_image(c):
        im=original_image(c)
        return cv2.resize(im,(c['w'],c['h']),interpolation=cv2.INTER_CUBIC)

    a.chips=scaled_chips
    a.image=scaled_image
    if args.fold is not None:
        source=ROOT/f'work/ae/ae_dfj_f{args.fold}.pt'
        assert source.is_file(),source
        original_build=a.build_det

        def from_joint(pretrained=True):
            model=original_build(False)
            state=torch.load(source,map_location='cpu',weights_only=True)
            model.load_state_dict({k:v.float() for k,v in state.items()})
            return model

        a.build_det=from_joint
        a.train(args.fold,iters=args.iters,bs=4,lr=3e-5,seed=20261024,tag=TAG)
    else:
        assert (OUT/f'work/ae/{TAG}_f{args.eval_fold}.pt').is_file()
        a.oof(tag=TAG,folds=(args.eval_fold,))

if __name__=='__main__':main()
