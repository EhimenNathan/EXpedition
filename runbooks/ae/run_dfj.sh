#!/bin/bash
cd "/home/nathan/Expedition Competition"
export DET_ARCH=dfine DET_JOINT=1 DET_QAUG=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in 0 1 2 3; do python3 -m koz3.ae_det_train --fold $f --iters 5000 --tag ae_dfj; done
python3 -m koz3.ae_det_train --oof --tag ae_dfj
echo AE_DFJ_DONE
