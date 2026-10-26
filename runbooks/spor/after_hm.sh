#!/bin/bash
cd "/home/nathan/Expedition Competition"
until grep -qE "EVAL_HM_BQ_DONE|Traceback" work/spor/eval_hm_bq.log; do sleep 60; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/sp_kclass_oof.py > work/spor/kclass.log 2>&1
REHEARSE_BLOCK=1 timeout 3000 python3 scripts/rehearse.py build/koz3_submission_v13_20261025.zip work/rehearsal_sp/input work/rehearsal_sp/gt.geojson > work/rehearse_v13.log 2>&1; echo EXIT $? >> work/rehearse_v13.log
echo AFTER_HM_DONE
