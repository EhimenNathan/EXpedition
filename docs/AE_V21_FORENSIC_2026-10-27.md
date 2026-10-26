# Aerial v21 forensic audit — 2026-10-27

## Public anchor

- v20 official score: **0.5018959557** (+0.0029970595 over v19b).
- The gain was isolated to Aerial. Public Aerial precision/recall became 0.102941/0.093333, reconstructing 7 TP, 122 FP, 68 FN.
- Every non-Aerial modality was unchanged. v20 remains the safe submission.

## Deployment checkpoint defect

The v20 archive contains only `inference/models/ae_det/w_0.pt`, byte-identical to `work/ae/ae_fcos_f2.pt`. Its own held-out fold had 0 TP / 39 FP / 4 FN for fortifications. This is a deployment weakness, but naive replacements did not generalize:

- all-data FCOS: rehearsal 0.5226 vs v20 0.5276; fortification F1 0.552 vs 0.651; rejected;
- best-validation fold-0 checkpoint: rehearsal 0.5155; fortification F1 0.410; rejected;
- fortification-only FCOS on untouched fold 0: best F1 0.214 vs multi-class baseline 0.383; rejected.

## Physical box audit

FCOS box width, height, aspect, area, and tile-edge clearance were reconstructed before centre-point conversion. A nested exact-metric grid over confidence, maximum physical area, and edge clearance failed: pooled fortification F1 0.2539 -> 0.2494. The attractive pooled oracle (0.2839) regressed fold 0 and is not deployable.

## Frozen visual verifier

A DINOv2-S crop verifier trained only on OOF proposals produced genuine geographic ranking signal:

- cross-fold AUC 0.783 (DINO only); combined representation AUC 0.797;
- fixed DINO-only rule (FCOS >=0.35 and verifier >=0.20) improved exact pooled fortification F1 0.2562 -> 0.2893, with no fold regression;
- integrated rehearsal regressed 0.5276 -> 0.5262 because crop extraction before tile fusion lost six known TPs.

The verifier is promising but **not submission-ready**. The next implementation must fuse FCOS detections first, select the highest-margin source chip for each fused world-coordinate candidate exactly as `ae_sheet_selfadapt.attach` does, then extract the verifier crop. Do not filter raw overlapping-tile detections before fusion. Repeat exact four-fold OOF and full-system rehearsal before packaging.

## Rejected external recommendations

- polygon area filters do not apply to emitted tiny centre circles; raw detector-box filters were tested directly and failed nested validation;
- LiDAR confirmation collapsed Aerial fortification F1;
- lowering the Or threshold is unsupported by geographic OOF;
- source-image isolation is already implemented for Ae, Or, and satellite; cross-image deduplication would violate organizer rules.
