# CPU-only pipeline from scratch (27 Oct)

Constraint: no GPU, no Codex; everything trained and evaluated in this 4-core CPU container (bf16 AMX).
Evaluation: official scorer (koz3.score), out of fold over koz3.sp_prep.site_folds (20 km clusters, 4 folds), every
threshold / rule chosen per class on the other folds (nested). Organiser rules respected: every source image is an
independent reference layer (no cross-image dedup; GT copied into each image footprint containing the object), mounds
inside settlements are not labelled (kurgan scene gate), WWII objects on mounds are separate objects.

## Satellite (132 regions, 175 images, layered GT 1,731 objects)

| step | honest score | what changed |
|---|---:|---|
| DINOv3-SAT dense tokens (8 m) + dilated-conv heads | 0.193 | frozen features too coarse: heat ≥ 0.1 at only 39 % of kurgans |
| + whole-scene DINOv3 hillfort localiser | 0.252 | hillfort F1 0.19 → 0.46; kurgan scene gate chosen in 4/4 folds |
| ResNet-18 CenterNet (2 m heat) alone | 0.226 | architecture 0.08 → 0.19, intact kurgans worse |
| map ensemble CNN + DINOv3 heads (equal weights) | 0.279 | damaged 0.341, architecture 0.247 |
| + framing prior (d_rel > 0.5 needs score ≥ 0.35) | 0.297 | chosen in 4/4 folds |
| + hillfort score = mean(scene classifier, localiser peak) | **0.326** | gate AUC 0.913; hillfort F1 0.585 (38 TP / 42 FP); damaged 0.385 |

Per class at 0.326: damaged kurgans 0.385, intact 0.143, hillforts 0.585, architecture 0.292, fortifications 0.000.
Reference: deployed v21 satellite ≈ 0.38 on its own (GPU) OOF, different folds and pre-layer GT → not directly
comparable; the CPU hillfort path (F1 0.585) is the component that clearly beats the deployed one.

## LiDAR (59 regions)
* LAS streamed → 0.5 m DTM (koz3.dtm, identical to inference) → relief image (multi-directional hillshade, local relief
  5 / 15 m); labels registered (≥ 90 % inside the DTM in every region).
* Hillfort localiser on whole-region relief tokens: hit 0.545 of 22 hillfort regions vs region centre 0.091; presence
  gate AUC 0.832.
* ResNet-18 CenterNet on relief (4 folds × 4,000 steps, bf16 CPU): **honest LiDAR 0.377** (damaged 0.459, intact 0.501,
  hillforts 0.400 with the relief localiser + presence gate, fortifications 0.070; the single LiDAR architecture label
  counts at weight 10 → ≈ 0.43 without it). Framing prior chosen in 4/4 folds. Deployed v21 LiDAR (GPU U-Net blend)
  ≈ 0.55–0.59 on its own folds → the CPU model is a candidate complementary member (blind sites), not a replacement.
* Rejected: class-aware ensemble weights for satellite (nested 0.299 < equal weights 0.326; choices flip between folds).
* Container restarts while idle: training checkpoints every 250 steps and resumes (verified kill → resume at step 250).

Scripts: experiments/claude/sp_dense/ (extract, heads, evaluate), experiments/claude/sp_cnn/ (cache, cache_li, train,
ensemble), experiments/claude/li_dense/ (extract, hillfort_tokens), experiments/claude/hf_scene_cls.py,
experiments/claude/sp_hillfort_dinov3_*.py.
