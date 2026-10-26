#!/bin/bash
cd "/home/nathan/Expedition Competition"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m koz3.sp_train --mod SpOr --oof --tag _bq > work/spor/oof_hm_bq.log 2>&1
cp work/spor/oof_hm_bq_cands.parquet work/spor/oof_det_hmbq_cands.parquet
for n in "dfine,rtdetr,hm,dfnodn,balq,balq3" "dfine,rtdetr,hmbq,dfnodn,balq,balq3" "dfine,rtdetr,hm,hmbq,dfnodn,balq,balq3"; do
  echo "== $n"; SP_NAMES=$n SP_MEMBERS=6 python3 scripts/sp_dual_oof.py 2>&1 | grep -E "tau 0.4|nested"
done
echo REDO_HM_DONE
