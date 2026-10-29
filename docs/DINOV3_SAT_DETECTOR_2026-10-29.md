# DINOv3-SAT detector experiment (29 October 2026)

## Question

Does the SAT-493M DINOv3 ViT-L/16 backbone add a geographically robust
satellite object detector to the deployed six-member ensemble?

## Protocol

- Exact existing four geographic folds, chips, labels, and official metric.
- Frozen SAT-493M backbone, intermediate transformer layers 8/16/24, spatial
  adapter/FPN, torchvision FCOS head initialized from COCO.
- 2500 training iterations per fold; low-threshold OOF export.
- Equal seventh-member fusion and a conservative rescue which leaves deployed
  predictions unchanged unless DINO corroborates a subthreshold cluster or is
  exceptionally confident and spatially unique.
- A predeclared fold-0 follow-up unfreezing the final four transformer blocks
  for 1500 iterations, initialized from the frozen model.

## Results

| Test | Result |
|---|---:|
| Frozen detector, standalone nested exact | 0.1371 |
| Existing six-member exact | 0.3664 |
| Equal seventh member | 0.3510 |
| Equal-member per-fold deltas | -0.0034, -0.0037, -0.0150, -0.0145 |
| Conservative rescue nested | 0.3664 (all folds choose no rescue) |
| Frozen fold-0 standalone best | 0.2075 |
| Last-four-block tuned fold-0 best | 0.1996 |

The frozen detector has real signal but is weaker and insufficiently
complementary.  Fine-tuning the final four blocks regresses the predeclared
fold-0 check.  The detector and every tested deployment path are rejected.

## Files

- `koz3/dinov3_sat_det.py`
- `scripts/sp_dsat_rescue_oof.py`

