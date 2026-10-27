# Data insights — 27 October 2026

Raw organiser data only (labels + LAS + optical rasters); no model outputs. Scripts: `experiments/claude/insight_*.py`.

## Labels: size and class structure (10,981 training labels)

- 5,000–10,000 m² (public recall 0.00): 112 training objects — satellite hillforts 28, satellite damaged kurgans 25,
  aerial damaged kurgans 17, LiDAR 19. Damaged kurgans > 5,000 m² are concentrated at one site: 041 Milovidovo holds
  65 of ~99 (giant ploughed mounds). The public zero likely covers a handful of objects of these two kinds.
- Hillforts: 118 labels in total (65 satellite, 30 LiDAR, 19 aerial, 4 ortho). Median area: satellite 9,206 m²,
  LiDAR 6,110 m², aerial 29,923 m².
- Damaged : intact = LiDAR 1371 : 782, satellite 1087 : 207, aerial 1162 : 587. Gnezdovo sites supply 39 % of all
  LiDAR kurgan labels.
- Public set (from the reconstructed v20 log): only ≈ 75 aerial objects in total (7 TP + 68 FN), so even a perfect
  aerial model is bounded.

## LiDAR weak sites: what the terrain under the labels looks like

Stacked 30 m local-relief model around every LiDAR kurgan label (0.25 m DTM from the LAS; figure
`figures/li_stacked_relief_by_site.png`, left→right: 102, 036, 101, 180, 011, 021):

| site | CNN recall (EXPERIMENTS.md) | stacked shape | centre / max relief | note |
|---|---|---|---|---|
| 102 Gnezdovo-Zaolshanskaya | 0.09–0.35 | **ring**, rim at 4–5 m | 0.00 / 0.35 m | looted mounds; label (median r 3.5 m) outlines the robber pit, the rim lies outside the label |
| 036 Timerevo | 0.41 | **ring**, rim at 2–3 m | 0.02 / 0.13 m | looted, low; rim inside the label |
| 101 Gnezdovo-Dneprovskaya | — | dome with central dip | 0.38 / 0.45 m | partly looted |
| 180 Gnezdovo-Lesnaya-2 | good | clean dome | 0.41 / 0.44 m | typical training mound |
| 011 Erentuna | 0.14–0.30 | large clean dome (~20 m wide) | 0.81 m | prominent yet missed — cause still open |
| 021 Novenkoe | 0.14–0.19 | small dome on a raised slope | 0.36 / 0.47 m | mounds on a terrace edge |

- Labels are registered to the LAS: the organisers' Li hillshade and our LAS hillshade correlate 0.89 (102) and
  0.79 (036) at zero offset, and every stack is centred on the label. A per-site "best shift" search had suggested a
  5.7 m offset at 102; the stack shows that this was the search sliding onto the rim of ring-shaped mounds.
- Point density does not explain the weak sites (docs/PHYSICS_GATES_2026-10-27.md).
- A physical ring matched filter does not recover the looted mounds across sites (template 036 → 102: recall 0.07;
  dome template 180 → 036 F1 0.41 vs ring template 102 → 036 F1 0.33).

## Implications

1. Two of the weakest LiDAR sites are dominated by looted (ring-shaped) mounds whose labels sit on the pit. For a
   peak-based decoder a rim response lands outside the label at 102. A model-side remedy to test on GPU: copy-paste
   augmentation that also carves central robber pits (depth 0.1–0.5 m, radius 0.3–0.6 × mound radius) into pasted
   mounds, and decoding by heatmap centre of mass instead of the strongest relief peak.
2. Erentuna (large, prominent, aligned domes) is a genuine model miss; a scale / regional appearance issue to inspect
   with the model's heatmaps.
3. The 5–10 k m² public zero is mostly two object kinds (satellite hillforts, giant ploughed kurgans), both rare.
