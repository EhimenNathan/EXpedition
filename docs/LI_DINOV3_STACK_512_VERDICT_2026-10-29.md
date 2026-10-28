# LiDAR DINOv3 presence stack at 512 px (2026-10-29)

## Verdict

Rejected. The frozen DINOv3-SAT relief localizer has independent location signal, but the deployed LiDAR hillfort mask is already a stronger site-presence signal and no cross-fitted rescue is safe.

## Evidence

- DINO map maximum presence AUC: **0.8571**.
- DINO scene-head presence AUC: **0.7095**.
- Deployed hillfort heat maximum AUC: **0.9143**.
- Deployed hillfort mask maximum AUC: **0.9476**.
- Six-signal cross-fitted logistic stack AUC: **0.8976**.
- Exact deployed LiDAR score: **0.58852**.
- Nested stack score: **0.58852**, delta **0.00000**; every fold selected the unchanged baseline.

No archive was built. The scheduled 1024 px run remains justified only as a localization-resolution test and will repeat the same exact stack evaluation.
