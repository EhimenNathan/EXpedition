#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -qE "LI_AUGX_S1_DONE|Traceback" work/li/train_augx_s1.log; do sleep 60; done
DUALS=0.05,0.15 python3 scripts/li_compare_oof.py \
  "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst" \
  "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst+work/li/oof_augx_s1.pkl" \
  "ens:work/li/oof_ens.pkl.zst*2+work/li/oof_aug.pkl.zst*2+work/li/oof_augx.pkl.zst+work/li/oof_augx_s1.pkl" \
  "ens:work/li/oof_augx.pkl.zst+work/li/oof_augx_s1.pkl" 2>&1 | grep -vi -E "warn|config.py|rasterize|multi-threading"
zstd -q -6 -T8 --rm work/li/oof_augx_s1.pkl -o work/li/oof_augx_s1.pkl.zst
echo EVAL_FINAL_DONE
