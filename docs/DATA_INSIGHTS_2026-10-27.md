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

## Drift: public composition differs from training

From the public structure file (file sizes) against the 2,132 training files:

| data type | median file train / public | share < 20 MB train / public |
|---|---|---|
| satellite | 4.2 / 191.5 MB | 0.87 / 0.45 |
| LiDAR LAS | 222 / 24 MB | 0.19 / 0.38 |
| aerial | 138 / 197 MB | 0.22 / 0.00 |

- Satellite: 17 of 30 public regions contain large scenes (≥ 20 MB) against 20 of 132 training regions (57 % vs 15 %).
  Large scenes are high-resolution (median GSD 0.15 m vs 0.74 m) and sparse in labels (5.1 vs 10.8 per km²).
  Inference resamples all satellite scenes to 0.5 m (`sp_or.py`, `read_scene(.., 0.5)`), so resolution drift is
  normalised, but the scene-type mix is not: OOF tuning weights regions equally and is dominated by screenshots.
  `scripts/sp_composition_oof.py` re-runs the deployed satellite decoding with regions importance-weighted to the
  public mix (nested, per-fold, unweighted result alongside).
- LiDAR: 9 of 13 public clouds are under 35 MB (small areas or sparse clouds); in training, small clouds are mostly the
  Swedish 1–3 pts/m² sites (Erentuna-like). The Erentuna scale test (above) matters more than its training weight.

## Uncertainty: how much of the score is the draw of regions

Monte Carlo (20,000 draws) of private sets with the public composition (17 large-scene + 13 screenshot satellite,
13 LiDAR, 4 aerial, 1 ortho, 2 MagGr, 1 EtGr, 1 TerFl regions resampled from training labels), each object detected
with the public per-data-type recall and false alarms at the public precision (v20 log). Model quality held fixed.

| private size | mean | SD | 90 % interval |
|---|---:|---:|---|
| ×0.5 | 0.528 | 0.045 | 0.444–0.591 |
| ×1 | 0.535 | 0.034 | 0.473–0.582 |
| ×2 | 0.536 | 0.025 | 0.491–0.572 |

Region draw alone moves the score by about ±0.055 at public size, more than every lever tested today and more than
the gap to the public leader. Kurgan classes dominate the variance (weight × per-class F1 SD: damaged 30 × 0.065,
intact 20 × 0.095); fortifications swing most per class (SD 0.14) at weight 5. The model is optimistic (mean 0.535 vs
public 0.502; no class confusion or site failures), so the true spread is larger. Choose finals by expected score
under resampling, never by public deltas below ~0.03.

## Small objects: emitted circle radius vs placement error

All 7,503 polygon labels; prediction = circle at the pole of inaccessibility plus Gaussian placement error σ,
official hit rule (centre inside the label, coverage ≥ 0.5 of the smaller shape)
(`experiments/claude/insight_radius_sim.py`).

- Radius 0.25–2 m changes hit rates by < 1 point for every class except architecture (+2 points at r = 0.25 m,
  48 % of ortho architecture labels are < 1 m wide inside) — not a lever.
- Placement error dominates for geophysics objects: 75 % of EtGr objects are < 1 m wide inside; hit rate 0.99 → 0.75
  at σ = 0.5 m → 0.55 at σ = 1 m. Kurgans and hillforts tolerate 1 m (≥ 0.97).
- The public ≤ 100 m² recall (0.63) equals the LiDAR / geophysics recall that makes up those objects; at the 0.5 m
  working grid the smallest-decile LiDAR kurgan still spans ~10 px. No small-object-specific deficit found.

## Geophysics objects: lightweight CPU detector (`experiments/claude/geo_objects_cpu.py`)

Pixel LightGBM (per-image normalised colour, multi-scale contrast and texture, method one-hot), leave-one-region-out
over 17 regions / 91 labelled images, per-image decoding, per-method thresholds nested, official metric:

| method | regions | nested F1 | TP / FP / FN |
|---|---:|---:|---|
| EtGr | 6 | 0.567 | 382 / 615 / 275 |
| MagGr | 12 | 0.276 | 120 / 1076 / 92 |
| TerFl | 2 | 0.234 | 65 / 625 / 112 |
| all object_poly | 17 | 0.435 | 541 / 1799 / 505 |

Overall it ties the deployed documented OOF (0.4297, different folds). Candidate use: EtGr maps and profiles only,
if the deployed EtGr-only OOF per region (212–217) is clearly lower. MagGr / TerFl stay with the deployed pipeline.
Integration needs a numpy rgb2lab (scikit-image may be absent on the platform).

## LiDAR physics-dome rescue (`experiments/claude/li_dome_detector.py`, `li_dome_rank.py`)

Candidates: σ = 5 m local-relief maxima on a 0.5 m DTM with 16-ray radial profiles (half-height radius, monotone
symmetry), site roughness and clutter density computed label-free. A LightGBM over these physical features, trained
leave-one-site-out on 13 sites, ranks candidates with AUC 0.877 (raw SNR 0.637); standalone pooled kurgan F1 0.430.

| site | physics ranker F1 (TP/FP/FN) | CNN (EXPERIMENTS.md) |
|---|---|---|
| 021 Novenkoe | 0.447 (33/111/26) | recall 0.14–0.19 |
| 036 Timerevo | 0.486 (154/185/233) | recall 0.41 |
| 011 Erentuna | 0.225 (56/744/13) | 26/69 found |
| 190 Muravlevo | 0.788 (81/5/41) | — |
| 017 Yushkovo | 0.785 (43/45/1) | — |
| 179/180/181 Lesnaya | 0.73 / 0.59 / 0.62 | recall 0.70–0.84 |
| 102 Zaolshanskaya | 0.077 | weak |

Not a replacement (CNN blend ≈ 0.57) but independent evidence that is strong exactly where the CNN is weak. Decisive
test: `scripts/li_dome_fusion_oof.py` adds ranked candidates only where the deployed CNN blend has no kurgan within
`gap` m, nested over the LiDAR folds with per-fold and per-site robust acceptance.
