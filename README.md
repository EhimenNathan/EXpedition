# KOZ-3 inference forensics

This private continuation repository contains the training, preprocessing,
inference, evaluation, rehearsal, and experiment code developed for the KOZ-3
multimodal archaeology competition by Claude and Codex. It excludes datasets,
labels, model weights, cached predictions, environments, and submission files.

## Main results

Electrical-survey maps, profiles, and depth layers are independently labelled
images.  The previous inference path removed coincident predictions across
those images and also executed mixed-family geophysics regions more than once.

The fixes:

- preserve predictions from every EtGr source image;
- queue each expert once per region with the union of its input-family paths;
- retain ordinary cross-image deduplication for modalities where it is valid.

On five site-disjoint geographic folds, EtGr layer preservation increased exact
`object_poly` F1 from **0.315012 to 0.429744**.  On the broad local submission
26-region full-system rehearsal, the combined fixes increased the official
score from **0.4635 to 0.4973**, increased `object_poly` F1 from **0.245 to
0.506**, improved `finds_points` F1 from **0.767 to 0.844**, and reduced runtime
from 310.4 to 250.7 seconds.  An EtGr-rich four-region smoke rehearsal scored
**0.5944**, with `object_poly` F1 **0.610**.

See [`docs/ETGR_LAYER_FIX_2026-10-26.md`](docs/ETGR_LAYER_FIX_2026-10-26.md)
for evidence and limitations.

The next verified deployment repair recovered an aerial-specific FCOS
fortification model that had been disabled together with a failed aerial
kurgan branch.  A layer-aware four-fold audit gives fortification F1
**0.253898** (114 TP / 752 FP / 294 FN).  On the authoritative 26-region
full-system rehearsal, enabling only aerial fortifications and preserving
satellite source-image layers raised the exact score from **0.4815 to 0.5276**
and fortification F1 from **0.0217 to 0.651**.  See
[`docs/AE_FORT_RECOVERY_2026-10-27.md`](docs/AE_FORT_RECOVERY_2026-10-27.md).

## Contents

- `koz3/`: shared data, training, scoring, geometry, and fusion modules.
- `scripts/`: preparation, OOF, build, validation, and platform rehearsal tools.
- `inference/`: submission entry point and modality experts.
- `experiments/codex/`: Codex CPU/GPU probes, Kaggle kernels, and rejected-model audits.
- `compute_metrics.py`: byte-verified official metric implementation.
- `configs/deployed_v19.json` and `configs/deployed_v20.json`: deployed
  thresholds and expert settings; weights omitted.
- `references/`: organiser baseline/simple source code, without model weights.
- `docs/EXPERIMENTS.md`: chronological experiment ledger, including rejected ideas.
- `docs/CONTINUATION_GUIDE.md`: setup, artifact map, validation gates, and next steps.
- `tests/test_etgr_layer_dedupe.py`: deployment invariant regression test.

Competition data, model weights, private labels, output archives, and organizer
assets are intentionally excluded.

Start with the [continuation guide](docs/CONTINUATION_GUIDE.md). Many scripts
reference the original workspace layout and expect data or cached OOF artifacts
under `data/`, `work/`, or `kaggle_import/`; those paths are documented rather
than silently replaced.

