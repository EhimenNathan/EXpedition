#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -qE "LI_SF1_DONE|Traceback" work/li/train_sf1.log; do sleep 60; done
python3 scripts/li_compare_oof.py work/li/oof_sf1.pkl "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst" "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_sf1.pkl" "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst+work/li/oof_sf1.pkl" 2>&1 | grep -vi -E "warn|config.py|rasterize|multi-thread"
zstd -q -6 -T8 --rm work/li/oof_sf1.pkl -o work/li/oof_sf1.pkl.zst
echo EVAL_SF1_DONE
