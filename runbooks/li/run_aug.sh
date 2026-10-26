#!/bin/bash
cd "/home/nathan/Expedition Competition"
for f in 0 1 2 3 4; do LI_AUG=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m koz3.li_train --fold $f --iters 12000 --tag _aug; done
LI_AUG=1 python3 -m koz3.li_train --oof --tag _aug
echo LI_AUG_DONE
