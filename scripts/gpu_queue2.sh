#!/usr/bin/env bash
# After the SpOr D-FINE run: D-FINE on Ae (partly annotated → chips within 300 m of labels) and on Or.
cd "$(dirname "$0")/.."
until grep -q "DET_DONE\|Traceback" work/spor/det_dfl.log; do sleep 60; done
until grep -q "chips;" data/ae_prep.log 2>/dev/null; do sleep 60; done
for f in 0 1 2 3; do python3 -m koz3.det_train --mod Ae --fold $f --iters 8000 --tag dfl >> work/ae/det_dfl.log 2>&1; done
python3 -m koz3.det_train --mod Ae --oof --tag dfl >> work/ae/det_dfl.log 2>&1; echo DET_AE_DONE >> work/ae/det_dfl.log
until grep -q OR_DONE data/or_prep.log 2>/dev/null; do sleep 60; done
for f in 0 1 2 3; do python3 -m koz3.det_train --mod Or --fold $f --iters 5000 --tag dfl >> work/or/det_dfl.log 2>&1; done
python3 -m koz3.det_train --mod Or --oof --tag dfl >> work/or/det_dfl.log 2>&1; echo DET_OR_DONE >> work/or/det_dfl.log
