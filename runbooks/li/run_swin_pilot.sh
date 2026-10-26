#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -qE "LI_AUGU_DONE|Traceback" work/li/train_augu.log; do sleep 120; done
export LI_ENC=tu-swin_tiny_patch4_window7_224 LI_AUG=1 LI_PASTE=1 RELIEF_EXTRA=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m koz3.li_train --fold 2 --iters 12000 --tag _swinx
echo SWIN_TRAINED
