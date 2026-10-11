## End-to-end GS-HOTA vs output delay, SoccerNet-GSR valid (58 clips)

| Configuration | 0 | 1 s | 5 s | 10 s | offline | Pitch HOTA |
|---|---|---|---|---|---|---|
| ByteTrack, colour team, raw PARSeq jersey | 26.4 | 28.1 | 30.3 | 31.1 | 32.2 | 63.4 |
| + McByte with shared camera-motion compensation | 26.5 | 28.3 | 30.5 | 31.2 | 32.0 | 64.1 |
| + jersey evidence gating | 27.8 | 29.8 | 32.4 | 34.0 | 35.2 | 64.1 |
| + pitch-space stitching | 28.7 | 30.5 | 33.0 | 34.6 | 36.2 | 66.2 |
| + (team, jersey) identity linking | 28.7 | 30.5 | 33.1 | 34.7 | 36.4 | 65.8 |
| + pose-guided jersey crops (final) | 29.5 | 31.6 | 34.3 | 35.9 | 37.9 | 66.0 |
| final, fine-tuned jersey reader (gate 0.9) | 30.1 | 32.1 | 34.9 | 36.7 | 38.5 | 66.0 |

## End-to-end GS-HOTA vs output delay, SoccerNet-GSR test (49 clips)

| Configuration | 0 | 1 s | 5 s | 10 s | offline | Pitch HOTA |
|---|---|---|---|---|---|---|
| final (pretrained jersey reader) | 34.6 | 37.1 | 39.9 | 42.1 | 43.6 | 66.1 |
| final, fine-tuned jersey reader | 34.7 | 37.2 | 39.6 | 41.6 | 43.0 | - |

## Component curves, valid (other attributes from ground truth)

| Component | 0 | 1 s | 5 s | 10 s | offline |
|---|---|---|---|---|---|
| jersey: raw PARSeq, tau 4 | 55.7 | 57.4 | 61.8 | 63.9 | 66.3 |
| jersey: gated (read conf >= 0.97) | 58.4 | - | 65.7 | - | 74.0 |
| jersey: gated + pose crops | 60.1 | - | 68.0 | - | 77.4 |
| jersey: gated + pose, fine-tuned reader | 59.6 | - | 67.0 | - | 76.5 |
| team: fit on first 1 s | 81.4 | 86.5 | 87.9 | 89.2 | 95.5 |
| team: online re-fit | 83.2 | 88.7 | 90.4 | 92.2 | 92.3 |

## Calibration-only GS-HOTA, valid (PnLCalib keyframes + optical-flow propagation)

| Policy | GS-HOTA | median error (m) | within 1 m | calibrator calls | propagation ms |
|---|---|---|---|---|---|
| every frame | 82.5 | 0.53 | 75.2% | 100.0% | - |
| keyframe every 5 | 82.2 | 0.55 | 74.8% | 19.5% | 4.5 |
| keyframe every 10 | 82.0 | 0.57 | 73.6% | 9.8% | 4.4 |
| keyframe every 25 | 79.6 | 0.70 | 66.5% | 3.9% | 4.4 |
| keyframe every 50 | 72.5 | 0.91 | 54.1% | 2.0% | 4.4 |
| first frame only | 29.4 | 4.92 | 15.5% | 0.1% | 4.4 |

## Error budget: ground truth with one factor degraded, valid

| Degradation | GS-HOTA |
|---|---|
| gt | 100.0 |
| no_jersey | 36.7 |
| no_team | 21.5 |
| team_side_swapped | 22.1 |
| roles_all_player | 88.2 |
| pitch_noise_0.5m | 94.3 |
| pitch_noise_1m | 75.5 |
| pitch_noise_2m | 39.3 |
| frame_bias_1m | 75.6 |
| frame_bias_2m | 39.4 |
| frame_bias_4m | 12.1 |
| id_fragment_250f | 63.4 |
| id_fragment_50f | 30.5 |
| drop_10pct | 90.0 |

## Oracle swaps on live end-to-end predictions, valid

| Swap in ground truth for | GS-HOTA live | GS-HOTA offline |
|---|---|---|
| pred | 26.4 | 32.2 |
| oracle_role | 27.9 | 33.6 |
| oracle_team | 30.8 | 35.0 |
| oracle_jersey | 39.0 | 44.1 |
| oracle_team+jersey | 50.2 | 50.2 |
| oracle_all_attrs | 60.5 | 60.6 |
| oracle_pitch | 30.3 | 37.1 |
| oracle_pitch+attrs | 72.0 | 72.1 |

## Image-space tracking on SoccerNet-GSR valid (cached detections)

| Tracker | HOTA | AssA | IDF1 | ID switches | tracker ms/frame |
|---|---|---|---|---|---|
| bm_bytetrack_low | 0.548 | 0.435 | 0.626 | 3607 | 3.45 |
| rf_sort | 0.504 | 0.363 | 0.532 | 12525 | 0.58 |
| rf_bytetrack | 0.584 | 0.489 | 0.684 | 3462 | 0.63 |
| rf_cbiou | 0.577 | 0.478 | 0.671 | 2427 | 1.06 |
| rf_ocsort | 0.460 | 0.351 | 0.527 | 4080 | 0.97 |
| rf_botsort_nocmc | 0.536 | 0.427 | 0.613 | 2808 | 1.02 |
| rf_botsort_cmc | 0.594 | 0.505 | 0.694 | 2025 | 42.31 |
| rf_mcbyte_nocmc | 0.540 | 0.432 | 0.620 | 2775 | 0.82 |
| rf_mcbyte_cmc | 0.597 | 0.510 | 0.701 | 2014 | 27.52 |
| rf_mcbyte_sharedcmc | 0.596 | 0.507 | 0.699 | 2015 | 12.20 |
