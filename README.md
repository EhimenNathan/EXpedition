# KOZ-3 inference forensics

This repository contains the deployment fixes and reproducible audits that
identified a layer-handling error in the KOZ-3 multimodal archaeology pipeline.

## Main result

Electrical-survey maps, profiles, and depth layers are independently labelled
images.  The previous inference path removed coincident predictions across
those images and also executed mixed-family geophysics regions more than once.

The fixes:

- preserve predictions from every EtGr source image;
- queue each expert once per region with the union of its input-family paths;
- retain ordinary cross-image deduplication for modalities where it is valid.

On five site-disjoint geographic folds, EtGr layer preservation increased exact
`object_poly` F1 from **0.315012 to 0.429744**.  On the broad local submission
rehearsal, the combined fixes increased the official score from **0.5599 to
0.5944**, increased `object_poly` F1 from **0.524 to 0.610**, and reduced the
runtime from 106 to 88 seconds.

See [`docs/ETGR_LAYER_FIX_2026-10-26.md`](docs/ETGR_LAYER_FIX_2026-10-26.md)
for evidence and limitations.

## Contents

- `inference/koz3_infer/run.py`: inference orchestration and final dedupe policy.
- `koz3/dedupe.py`: region-level cross-image duplicate removal.
- `audits/annotation_multiplicity_audit.py`: annotation multiplicity audit.
- `audits/geo_etgr_multiplicity_oof.py`: five-fold exact-metric evaluation.
- `tests/test_etgr_layer_dedupe.py`: deployment invariant regression test.

Competition data, model weights, private labels, output archives, and organizer
assets are intentionally excluded.

