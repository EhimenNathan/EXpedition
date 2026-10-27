# Physics-based gates — 27 October 2026

Two physically motivated levers were tested on CPU with the organisers' training data before spending GPU time.

## 1. LiDAR point density — terrain hypothesis rejected

Training clouds differ widely in density: 1–76 ground points per occupied 1 m² cell (`docs/lidar_density.csv`;
the bounding-box figure underestimates irregular footprints). Two of the three known blind sites are sparse
(011 Erentuna 2 /m², 102 Gnezdovo-Zaolshanskaya 3 /m²), which suggested that sparse sampling degrades the DTM.

Controlled experiment (036 Timerevo, 18 /m², 387 LiDAR kurgans): the same cloud thinned with `dtm.build(keep=…)`.

| density | peak-relief corr. with full | median abs. error | kurgans with a ≥15 cm relief peak |
|---|---:|---:|---:|
| 18 /m² (reference) | 1 | 0 | 0.729 |
| 4 /m² | 0.999 | 0.008 m | 0.718 |
| 2 /m² | 0.998 | 0.012 m | 0.718 |
| 1 /m² | 0.992 | 0.016 m | 0.700 |
| 0.5 /m² | 0.984 | 0.021 m | 0.711 |

Gaussian normalised-convolution interpolation instead of nearest-cell filling was no better (0.5 /m²: error 0.029 m,
kurgans with a peak 0.685) and was dropped. Mound relief survives thinning; density does not explain blind sites by
itself. `scripts/li_density_gate.py` still checks for a model-side density effect on the deployed OOF; a thinning
augmentation run is justified only if that gate fires.

## 2. Satellite illumination consistency — weak but real signal

`koz3/shading.py`: brightness-centroid dipole of each candidate and its agreement with the leave-one-out sun azimuth
of the image. Synthetic check: mounds point to the sun, pits away (`tests/test_shading.py`).

On labelled kurgans of 19 single-image satellite scenes the dipoles share one direction far more than random
locations: p < 0.01 in 9/19 scenes, p < 0.05 in 12/19.

Proxy discrimination test (11,207 LoG blob candidates, 171 inside kurgans, leave-one-scene-out logistic regression):

| features | pooled AUC | mean per-scene AUC |
|---|---:|---:|
| appearance only (LoG response, scale, polarity) | 0.540 | 0.588 |
| + shading, reference = top-30 % blobs (unreliable) | 0.548 | 0.556 |
| + shading, reference = other true kurgans | 0.594 | 0.613 |

The signal depends on reliable reference detections; confident CNN detections sit between the two references.
The decisive test is `scripts/sp_shading_oof.py` on the deployed v20 satellite OOF (nested λ, per-fold check,
leave-largest-site-out k*, bootstrap). Expected effect if it passes: small (+0.002 to +0.008), not a step change.
