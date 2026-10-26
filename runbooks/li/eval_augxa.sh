#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -qE "LI_AUGXA_DONE|Traceback" work/li/train_augxa.log; do sleep 120; done
python3 scripts/li_compare_oof.py work/li/oof_augxa.pkl "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst" "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augxa.pkl" "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst+work/li/oof_augxa.pkl" 2>&1 | grep -vi -E "warn|config.py|rasterize|multi-thread"
echo EVAL_AUGXA_DONE
