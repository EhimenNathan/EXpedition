#!/usr/bin/env bash
# Sequential GPU queue after the satellite run: LiDAR (5 folds, their folds) → Ae (4 folds) → Or (4 folds).
cd "$(dirname "$0")/.."
until grep -q "SP_DONE\|Traceback" work/spor/train.log 2>/dev/null; do sleep 30; done
until grep -q LAS_DONE data/las_prep.log; do sleep 60; done
python3 -m koz3.li_train --cache >> work/li/train.log 2>&1
for f in 0 1 2 3 4; do python3 -m koz3.li_train --fold $f --iters 12000 >> work/li/train.log 2>&1; done
python3 -m koz3.li_train --oof >> work/li/train.log 2>&1; echo LI_DONE >> work/li/train.log
until [ -s data/ae_prep.log ] && grep -q "chips;" data/ae_prep.log; do sleep 60; done
for f in 0 1 2 3; do python3 -m koz3.sp_train --mod Ae --fold $f --iters 10000 >> work/ae/train.log 2>&1; done
python3 -m koz3.sp_train --mod Ae --oof >> work/ae/train.log 2>&1; echo AE_DONE >> work/ae/train.log
until grep -q OR_DONE data/or_prep.log 2>/dev/null; do sleep 60; done
for f in 0 1 2 3; do python3 -m koz3.sp_train --mod Or --fold $f --iters 6000 >> work/or/train.log 2>&1; done
python3 -m koz3.sp_train --mod Or --oof >> work/or/train.log 2>&1; echo OR_TRAIN_DONE >> work/or/train.log
