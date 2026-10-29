# Paper triage and high-resolution detector experiments (2026-10-29)

## New experiments

1. **Satellite YOLO11m-P2**: a real stride-4 detection head at 1024 px, motivated by HSF-YOLO, MambaIR-YOLO and A2-Det.  It is evaluated as a seventh member on the exact deployed geographic folds.
2. **Trusted-label kurgan P2**: binary proposal model trained only on regions where kurgans were annotated.  This addresses partial-label imbalance rather than merely changing loss weights.
3. **DINOv3-SAT architecture localizer**: the 512 px cross-fitted map has token AUC 0.979 and localizes 23/29 positive scenes.  A fold-nonregressive selector gains +0.00387 satellite score, but the exact v26 1024 representation selects no rescue; not packaged.

## Partial annotation finding

Satellite labels span 132 regions, but 96 regions contain only one labeled family.  Region counts with labels are:

- damaged kurgans: 60
- intact kurgans: 30
- hillforts: 49
- architecture: 28
- fortifications: 5

Only 6 hillfort regions overlap damaged-kurgan labels and only 3 architecture regions overlap them.  Ordinary multi-class training consequently treats many objects outside the region's annotation purpose as background.  The trusted-label pilot uses within-kurgan-region background while avoiding unrelated single-family regions.

## Technique triage

- **HSF-YOLO / MambaIR-YOLO / A2-Det**: their common actionable mechanism is a high-resolution P2/detail path.  This is the active prospective test.
- **DyPerceiver-Det / Cascade R-CNN / rotated boxes**: primarily improve oriented-box localization.  KOZ-3 scores point containment, so their main advantage does not address the measured error.
- **Swin-L / Satlas**: the full-system Satlas member already regressed rehearsal 0.5367 to 0.5289.  Parameter count alone is not a useful lever.
- **Class imbalance**: site balance, focal-style losses and per-class thresholds already exist.  Partial-label-aware supervision is the untested structural fix.
- **NMS**: physical-radius grids were already tested for satellite and LiDAR; current radii won geographically nested evaluation.
- **Google Earth augmentation**: Google Maps/Earth terms do not permit building or copying a training dataset in the required transferable competition archive.  It is excluded.
- **NVIDIA NVDINOv2**: the available TAO checkpoint is CC-BY-NC-SA and conflicts with competition rule 13.4.  C-RADIO is license-compatible but is general-image pretrained and lower priority than the satellite-specific DINOv3-SAT model already tested.
- **GAN/synthetic relief**: ring-aware LiDAR augmentation completed and failed the geographic acceptance gate; repeating synthetic-shape generation is not justified.

## Acceptance rule

Any P2 result must improve exact official scoring on all four site-disjoint outer folds, pass the robust site report, and then improve a full v26 rehearsal before packaging or submission.
