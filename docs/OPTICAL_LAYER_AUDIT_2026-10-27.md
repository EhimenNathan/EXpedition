# Optical source-layer audit — 27 October 2026

## Contract

The organisers clarified that every source image is an independent reference
layer. If the same physical object appears in two satellite or aerial images,
it is represented by two ground-truth objects. Images must remain in their own
coordinate systems and predictions must not be fused or deduplicated across
source images.

## Deployment defect

The v19b satellite expert accumulated the detector outputs from every satellite
image in a region and called `sp_fuse.fuse` once. This could collapse detections
from multiple source images into one output. Region-level duplicate removal
could collapse further detections because it did not know the source layer.

The candidate correction:

1. runs member fusion independently for every satellite source image;
2. stores the exact filename in `properties.source_layer`;
3. includes `source_layer` in the region-dedupe grouping key, preserving the
   existing within-image duplicate removal while preventing cross-image loss;
4. leaves the single-image execution path unchanged.

`scripts/build_optical_layer_gt.py` constructs a layer-aware rehearsal target by
copying satellite labels into every source image footprint containing their
representative point. It is an audit utility; it does not modify competition
training data.

## Exact rehearsal

On the 26-region deployment rehearsal, satellite GT expanded from 451 to 507
objects (+56) across four multi-image regions. With the official metric:

| system | final score | damaged F1 | hillfort F1 | architecture F1 |
|---|---:|---:|---:|---:|
| v19b pooled across images | 0.4891 | 0.531 | 0.525 | 0.770 |
| source-layer candidate | 0.5037 | 0.534 | 0.561 | 0.854 |

Within the four multi-image regions, the score rose 0.26873 to 0.33040. The
candidate recovered four damaged kurgans, one hillfort and ten architecture
objects while reducing damaged and architecture false positives. Geophysics,
LiDAR and finds counts were unchanged. Packaged inference completed in 238 s,
wrote 2,732 features once atomically, and stayed inside all output limits.

## Decision

This is a correct data-contract repair and a substantial effect on multi-image
regions. Its expected public effect is much smaller because the recovered
public composition indicates only one satellite region with multiple images.
The prior source-aware geographic OOF gain was small (damaged F1 0.39744 to
0.40286; intact 0.30108 to 0.30632). Keep the code and evidence, but do not
submit solely on the rehearsal gain unless public multi-image exposure is
confirmed or another validated gain can be bundled with it.
