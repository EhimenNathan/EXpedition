#!/bin/bash
cd "/home/nathan/Expedition Competition"
export LI_AUG=1 LI_PASTE=1 RELIEF_EXTRA=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in 0 1 2 3 4; do python3 -m koz3.li_train --fold $f --iters 12000 --tag _augx_s1 --seed 1; done
python3 -m koz3.li_train --oof_dtm --tag _augx_s1
echo LI_AUGX_S1_DONE
