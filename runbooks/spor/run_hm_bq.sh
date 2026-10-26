#!/bin/bash
cd "/home/nathan/Expedition Competition"
export SP_SITE_BAL=1 SP_QAUG=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in 0 1 2 3; do python3 -m koz3.sp_train --mod SpOr --fold $f --iters 10000 --tag _bq --seed 2; done
python3 -m koz3.sp_train --mod SpOr --oof --tag _bq
echo SP_HM_BQ_DONE
