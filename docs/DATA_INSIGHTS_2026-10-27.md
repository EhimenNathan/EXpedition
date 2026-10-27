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

## LiDAR signal-to-clutter by site (the model's own relief scales)

Local relief L_σ = DTM − Gaussian(DTM, σ) at 0.5 m, as in `koz3/relief.py`. Signal = median of the maximum L_σ inside
each kurgan label; clutter = std of L_σ on background (> 10 m from any label); look-alikes = background local maxima
reaching half the median mound signal (`experiments/claude/insight_li_snr.py`).

| site | known kurgan recall | SNR σ=2 m | SNR σ=5 m | look-alikes/km² (σ=5 m) |
|---|---|---:|---:|---:|
| 180 Gnezdovo-Lesnaya-2 | 0.70–0.84 | 11.2 | 6.6 | 0 |
| 179 Gnezdovo-Lesnaya-1 | 0.70–0.84 | 8.3 | 6.9 | 1,314 |
| 181 Gnezdovo-Lesnaya-3 | 0.70–0.84 | 5.6 | 4.4 | 1,151 |
| 021 Novenkoe | 0.14–0.19 | 2.2 | **1.2** | 2,421 |
| 102 Gnezdovo-Zaolshanskaya | 0.09–0.35 | 3.0 | **1.7** | 3,343 |
| 036 Timerevo | 0.41 | 2.8 | **1.9** | 3,027 |
| 011 Erentuna | 0.14–0.30 | 7.2 | **4.9** | **539** |

Other sites (recall not recorded): 004 Borovichi-1 4.6, 013 Nora-Vinnon 2.4, 017 Yushkovo 8.1, 028 Vyshegzha-2 2.3,
030 Esbyu 4.1, 101 Gnezdovo-Dneprovskaya 3.0, 190 Muravlevo 3.5.

- Three of the four weak sites are physics-limited: mound relief is only 1.2–1.9× the background roughness and
  natural look-alikes are dense. This is a recall ceiling, not a training defect; do not target it.
- Erentuna is the exception: prominent mounds (0.79 m at σ=5 m), flat ground, few look-alikes — a genuine model miss.
  Its mounds are the largest in the set (median diameter 12 m, p90 18 m) against 6–10 m at the dominant training
  sites, beyond the ×0.7–1.45 zoom augmentation. GPU test: zoom augmentation to ×2.2 and/or an additional half-
  resolution inference pass, judged on held-out folds containing 011, 004, 013, 016, 018, 104.

## Satellite hillfort placement without image evidence — rejected

63 satellite hillforts in screenshots cover a median 1.1 % of the scene (normalised radius ≈ 0.12) with centroids a
median 0.23 half-sizes from the centre; the centre is inside 30 %. A greedy point pattern learned on the other regions
hits only 0.125 new hillforts per extra emitted point on held-out regions (in scenes known to contain one), below the
≈ 0.14 break-even before gate errors. Extra blind points lose; placement needs image evidence (the deployed snap).
Hillfort polygons are solid (median solidity 0.98, no holes), so the misses are offsets, not hollow shapes.
