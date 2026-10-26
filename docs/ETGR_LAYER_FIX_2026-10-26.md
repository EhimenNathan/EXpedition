# EtGr layer preservation and geophysics scheduling fix

## Finding

The competition contract defines every electrical-survey map, profile, and
depth layer as an independent reference image.  Coincident objects on two
images are therefore two distinct targets and must produce two predictions,
each carrying its source image.

The inference code violated this contract in two places:

1. Final region deduplication grouped `object_poly` predictions by region and
   modality, then removed coincident predictions from different EtGr images.
2. The orchestrator queued one geophysics region once for every family present
   (`maggr`, `etgr`, and `terfl`), although `geo.predict` scans all geophysics
   inputs in one call.  A mixed-family region could consequently run several
   times and produce duplicate predictions for the same image.

The corrected invariants are:

- queue each expert once per region, with the union of its family paths;
- retain separate EtGr predictions from separate source images;
- keep ordinary cross-image deduplication for modalities whose labels describe
  one physical object rather than one object per source layer.

## Geographic OOF evidence

The exact object-matching metric was evaluated site-disjoint over five folds.

| Decoder | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| Existing region dedupe | 0.315012 | 288 | 909 | 798 |
| Preserve EtGr images | **0.429744** | **458** | 1175 | **628** |

Absolute F1 gain: **+0.114732**.  Four of five folds improved, and nested
selection chose layer preservation in all five outer folds.  Fold deltas were
`+0.059133, -0.001909, +0.108575, +0.178852, +0.212518`.

This is a component result.  Its public-score effect depends on the hidden
class and modality mix, so it must not be presented as a public score forecast.

## Paired full-system rehearsal

Both archives were run on the same 26-region, multi-modal rehearsal with the
official metric implementation.

| Archive | Score | Features | Runtime | object_poly F1 | finds_points F1 |
|---|---:|---:|---:|---:|---:|
| v18b | 0.4635 | 4,300 | 310.4 s | 0.245 | 0.767 |
| v19b corrected | **0.4973** | **2,723** | **250.7 s** | **0.506** | **0.844** |

The exact paired full-system gain is **+0.0338**.  The corrected run completed
with a single atomic output write, 2,723 objects, and a 3.7 MB GeoJSON result.
The changes do not alter polygon construction, so the established vertex limit
remains unchanged.

On an EtGr-rich four-region smoke rehearsal, the corrected archive scored
**0.5944**, with `object_poly` F1 **0.610**, and completed in 87.9 seconds.

## Regression coverage

`Codex Work/test_etgr_layer_dedupe.py` checks all three required behaviors:

- two coincident EtGr objects from different images remain two predictions;
- two coincident ordinary imagery predictions are deduplicated;
- a region containing EtGr and TerFl is queued exactly once.

## Official-log autopsy

- v18b (`0.4890562536`) is behaviorally identical to v14 in all reported
  modality precision and recall values.
- v18a (`0.4867759009`) is behaviorally identical to v17.  Its regression is
  isolated primarily to satellite precision/recall (`0.429091/0.625995` versus
  v18b's `0.434234/0.639257`).
- Contour diagnostics printed by the evaluator do not enter `final_score`.
  Polygon shrinking based only on those diagnostics is therefore unsupported.

## Data update audit

The local 101/102 Gnezdovo LiDAR caches were prepared after the organizer's
updated source timestamps.  No evidence currently indicates that the active
LiDAR pipeline was trained from the superseded maps.  Raw 102 LAS was removed
after preprocessing to conserve disk, while its prepared cache remains.

