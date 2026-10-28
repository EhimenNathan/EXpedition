#!/bin/bash
set -euo pipefail
ROOT="/home/nathan/Expedition Competition/Codex Work/dinov3_wt"
RING="/home/nathan/Expedition Competition/work/li"
cd "$ROOT"
while ! grep -q "WEIGHTS_AUGXR_DONE" "$RING/weights_augxr.log" 2>/dev/null; do
  sleep 60
done
LI_DINO_SIDE=1024 python3 experiments/codex/li_hillfort_dinov3_tokens.py work/hf_dinov3/li_tokens1024.npz
python3 experiments/codex/li_hillfort_dinov3_eval.py work/hf_dinov3/li_tokens1024.npz work/hf_dinov3/li_loc_points_oof_1024.csv
nice -n 5 python3 experiments/codex/li_hillfort_dinov3_exact_oof.py work/hf_dinov3/li_loc_points_oof_1024.csv
nice -n 5 python3 experiments/codex/li_hillfort_dinov3_stack_oof.py work/hf_dinov3/li_loc_points_oof_1024.csv
echo LI_DINO1024_DONE
