"""Independent Ae-only fine-tune from Claude's completed joint D-FINE weights.

Prepare now; run only when the local GPU is genuinely free. Reads original
weights and chips, writes solely under Codex Work/ae_yolo_pilot/dfine_ft.
Training folds retain their original site split; no held-out labels are used.
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
OUT=ROOT/'Codex Work/ae_yolo_pilot/dfine_ft'
sys.path.insert(0,str(ROOT))


def assert_gpu_clear():
    if os.environ.get('KOZ3_ISOLATED_GPU')=='1':
        return
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader'],text=True)
    users=[]
    for line in output.splitlines():
        pid,mem=(x.strip() for x in line.split(',',1))
        if int(mem.split()[0])>=1000 and int(pid)!=os.getpid():
            users.append((pid,mem))
    if users:
        raise RuntimeError(f'GPU in use by other jobs: {users}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,choices=range(4))
    ap.add_argument('--oof',action='store_true')
    ap.add_argument('--eval-fold',type=int,choices=range(4))
    ap.add_argument('--iters',type=int,default=1200)
    args=ap.parse_args()
    assert sum((args.oof,args.fold is not None,args.eval_fold is not None))==1, \
        'Choose one of --fold, --eval-fold or --oof'
    assert_gpu_clear()
    import torch
    from koz3 import ae_det_train as a
    config=ROOT/'hf_dfine_config'
    if config.is_dir():a.DFINE=str(config)
    a.ROOT=str(OUT)
    if args.fold is not None:
        source=ROOT/f'work/ae/ae_dfj_f{args.fold}.pt'
        assert source.is_file(),source
        original=a.build_det

        def from_joint(pretrained=True):
            model=original(False)
            state=torch.load(source,map_location='cpu',weights_only=True)
            model.load_state_dict({k:v.float() for k,v in state.items()})
            return model

        a.build_det=from_joint
        a.train(args.fold,iters=args.iters,bs=4,lr=3e-5,
                seed=20261024,tag='codex_ae_ft')
    elif args.eval_fold is not None:
        assert (OUT/f'work/ae/codex_ae_ft_f{args.eval_fold}.pt').is_file()
        a.oof(tag='codex_ae_ft',folds=(args.eval_fold,))
    else:
        for fold in range(4):
            assert (OUT/f'work/ae/codex_ae_ft_f{fold}.pt').is_file()
        a.oof(tag='codex_ae_ft')


if __name__=='__main__':main()
