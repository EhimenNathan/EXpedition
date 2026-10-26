#!/usr/bin/env bash
# D-FINE-X (Objects365) with the previous team's proven train_sp_det.py recipe (only the checkpoint changes).
cd "$(dirname "$0")/../kaggle_import/code"
until grep -q "OR_TRAIN_DONE\|Traceback" ../../work/or/train.log 2>/dev/null; do sleep 60; done
while pgrep -f "^python3 scripts/prep_sp[.]py" >/dev/null; do sleep 60; done
export HF_HUB_OFFLINE=1
python3 scripts/train_sp_det.py --data work/sp --folds 0,1 --ckpt ustc-community/dfine-xlarge-obj365 --iters 8000 --bs 8 --workers 4 --tag sp_dfine_x --outroot work/sp_runs > ../dfine_x_01.log 2>&1
python3 scripts/train_sp_det.py --data work/sp --folds 2,3 --ckpt ustc-community/dfine-xlarge-obj365 --iters 8000 --bs 8 --workers 4 --tag sp_dfine_x --outroot work/sp_runs > ../dfine_x_23.log 2>&1
echo DFINE_X_DONE >> ../dfine_x_23.log
