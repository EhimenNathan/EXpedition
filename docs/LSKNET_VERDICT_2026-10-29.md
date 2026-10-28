# LSKNet-T satellite verdict — 2026-10-29

## Prospective question

Can the official lightweight LSKNet-T remote-sensing backbone add independent
hillfort location or scene-presence signal to the accepted v26 DINOv3-SAT
stack, while reducing archive and latency cost?

## Method

- Official LSKNet-T DOTA 1.0 checkpoint (`lsk_t_fpn_1x_dota_le90`).
- Frozen stride-16 features from the 1024-pixel whole-scene view.
- Hillfort probes trained out of fold by geographic site fold, using the same
  labels, grids and fixed linear-probe family as the DINOv3 localizer.
- A second cross-fitted scene-presence probe used pooled stride-32 features.
- The scene score was then added to the already accepted deployed-scene +
  DINO-map + DINO-scene logistic stack under the same geographic folds.

## Result

| Test | LSKNet-T | Reference |
|---|---:|---:|
| Fixed localizer hit rate on 55 positive scenes | 0.127 | DINOv3-SAT 0.691; scene center 0.345 |
| Nested localizer hit rate | 0.109 | — |
| Scene-presence AUC | 0.657 | accepted three-source stack 0.9455 |
| Stack AUC after adding LSKNet | 0.9425 | accepted stack 0.9455 |

Per-fold AUC also did not show a consistent complement: adding LSKNet changed
`[.9295, .9965, .9185, .9750]` to
`[.9263, .9965, .9222, .9750]`.

## Decision

Reject. LSKNet-T is small and fast, but its DOTA/ImageNet representation does
not localize archaeological hillforts and weakens the accepted presence stack.
Do not package it or spend a public submission on it. Presto, Prithvi and DOFA
were not run because their temporal/multispectral input contract does not match
the competition's high-resolution RGB screenshots; GeoSAM's boundary objective
does not affect the official final score.
