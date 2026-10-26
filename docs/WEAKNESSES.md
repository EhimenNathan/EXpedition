# KOZ-3 — current weaknesses (24 Oct, v11 = 0.44032 public; v12 staged)

Evidence: platform logs v9 / v11 (per-type P/R, size recall), nested OOF studies (docs/EXPERIMENTS.md).
Weighted loss = class weight × (1 − F1), simulator per class for the v12 configuration.

## A. By class (where the points are)
| # | Class (weight) | F1 (sim, v12 cfg) | Points lost | Main cause |
|---|---|---|---|---|
| 1 | Damaged kurgans (30) | 0.41 | ~17.7 | Ae labels all missed; satellite FPs; LiDAR blind sites |
| 2 | Intact kurgans (20) | 0.46 | ~10.8 | class confusion with damaged; few intact training labels on satellite (207 vs 1 087) |
| 3 | Hillforts (15) | 0.41 | ~8.8 | localisation (scene centre inside the hillfort in only ~40 %); big scenes rely on weak detectors |
| 4 | Architecture (10) | 0.27 | ~7.3 | Or recall 0.24; satellite architecture weak; only 4 Or training sites (one cluster) |
| 5 | Objects (10) | ~0.42 (deployed blend) | ~5.8 | magnetometry objects low precision; thermography has 2 training regions |
| 6 | Fortifications (5) | 0.08 | ~4.6 | LiDAR trenches mostly undetected (23 / 264); Ae fortifications off with Ae |
| 7 | Finds (10) | 0.66 | ~3.4 | near the physics + GBM optimum (0.736 pooled OOF) |

## B. By data type
| # | Data type | Public P / R (v11) | Weakness | Evidence |
|---|---|---|---|---|
| 8 | Ae (4 val regions, ~21 % of damaged-kurgan labels) | 0 / 0 (off) | kurgans on 1940s film undetectable at unseen sites | best held-out F1 0.11–0.12 at ~5 % precision (5 detector families incl. joint SpOr+Ae D-FINE); public precision ~3.5× worse than OOF |
| 9 | Satellite (30 regions) | 0.38 / 0.60 | false positives on empty ground far from the site; class confusion | precision falls 0.38 → 0.05 from scene centre to corners; 97 cross-class landings |
| 10 | LiDAR (13 regions) | 0.60 / 0.64 | whole sites blind (Novenkoe 0.14–0.17, Erentuna 0.26–0.30, Gnezdovo-Zaolsh. 0.09–0.35) | same model 93–100 % recall when the site is in training → site-specific appearance, not capacity |
| 11 | LiDAR misses | — | 598 of 870 missed kurgans get heat < 0.05 (not detectable by any threshold) | li_fn_analysis |
| 12 | Orthophoto (1 region) | 0.71 / 0.24 | low recall; trained on 4 Caucasus sites (LORO 0.22) | public precision far above OOF (0.21) — can't tune to one region |
| 13 | Geophysics (4 regions) | 0.68 / 0.72 | magnetometry objects ~10 % precision before re-threshold; thermography 2 regions | per-method study |

## C. By object size (v11 public recall)
| # | Size | Recall | Likely objects |
|---|---|---|---|
| 14 | ≤ 100 m² | 0.60 | small kurgans, finds |
| 15 | 500–1 000 m² | 0.41 | large kurgans / small hillfort parts |
| 16 | 5 000–10 000 m² | **0.00** | small hillforts, large fortification systems |
| 17 | > 10 000 m² | 0.29 | hillforts (HD95 151 m: our point often outside) |

## D. Methodological / systemic
| # | Weakness | Consequence |
|---|---|---|
| 18 | Tiny validation per type (4 Ae, 2 MagGr, 1 EtGr, 1 TerFl, 1 Or) | public score is noisy; tuning to it overfits the closed set |
| 19 | Simulator mis-calibrated for Ae precision (OOF 15 % vs public 4.5 %) and blind to twins / thermography | Ae / twin decisions need platform evidence |
| 20 | Large seed-to-seed variance (LiDAR singles 0.545–0.569) | single-model results can mislead; ensembles needed |
| 21 | Training imbalance: Gnezdovo dominates LiDAR; dense cemeteries dominate satellite; few intact / fortification labels | site balance helped (+0.012–0.016) but not enough |
| 22 | Correlated satellite detectors (D-FINE / RT-DETR family) | each extra member ≤ +0.008 |
| 23 | Labels exist only near the site (framing) | true but unlabelled objects score as FP — exploited by the framing prior, not fixable |
| 24 | Point placement for large objects | representative point must fall inside GT; centre / peak placement misses irregular hillforts |
| 25 | No co-located modalities in validation for Ae / Or / most satellite regions | cross-modal confirmation helps only where LiDAR + satellite overlap |

## E. What is NOT a weakness (headroom we already use)
* Runtime 10.5 / 30 min — room for TTA / more members. Packaging, atomic write, platform imports: verified.
* Post-processing: 6 + techniques tested nested (positions, calibration, stacking, NMS, sub-pixel, morphology) — saturated.
