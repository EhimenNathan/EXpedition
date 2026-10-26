#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -qE "LI_AUGU_DONE|Traceback" work/li/train_augu.log; do sleep 60; done
python3 scripts/li_compare_oof.py work/li/oof_augu.pkl \
  "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augx.pkl.zst+work/li/oof_augu.pkl" \
  "ens:work/li/oof_ens.pkl.zst+work/li/oof_aug.pkl.zst+work/li/oof_augu.pkl" 2>&1 | grep -vi -E "warn|config.py|rasterize|multi-threading"
zstd -q -6 -T8 --rm work/li/oof_augu.pkl -o work/li/oof_augu.pkl.zst
echo EVAL_AUGU_DONE
