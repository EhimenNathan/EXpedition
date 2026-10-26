#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -q LI_AUG_DONE work/li/train_aug.log; do sleep 60; done
export LI_AUG=1 LI_PASTE=1 RELIEF_EXTRA=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in 0 1 2 3 4; do python3 -m koz3.li_train --fold $f --iters 12000 --tag _augx; done
python3 -m koz3.li_train --oof_dtm --tag _augx
echo LI_AUGX_DONE
