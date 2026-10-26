#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -qE "SP_HM_BQ_DONE|Traceback" work/spor/train_hm_bq.log; do sleep 60; done
cp work/spor/oof_hm_bq_cands.parquet work/spor/oof_det_hmbq_cands.parquet
for n in "dfine,rtdetr,hm,dfnodn,balq,balq3" "dfine,rtdetr,hmbq,dfnodn,balq,balq3" "dfine,rtdetr,hm,hmbq,dfnodn,balq,balq3"; do
  echo "== $n"; SP_NAMES=$n SP_MEMBERS=6 python3 scripts/sp_dual_oof.py 2>&1 | grep -E "tau 0.4|nested"
done
echo EVAL_HM_BQ_DONE
