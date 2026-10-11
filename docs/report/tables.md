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

## Clip-level bootstrap, test (49 clips, 2000 resamples)

| Set | GS-HOTA | 95% CI |
|---|---|---|
| live | 34.57 | [31.32, 37.79] |
| d1s | 37.09 | [33.69, 40.56] |
| d5s | 39.89 | [36.04, 43.77] |
| d10s | 42.13 | [37.98, 46.30] |
| offline | 43.62 | [39.33, 47.78] |
| ft_live | 34.67 | [31.41, 37.95] |
| ft_offline | 43.03 | [38.93, 46.97] |

| Paired difference | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| d1s - live | +2.51 | [+2.13, +2.93] | 0.000 |
| d5s - live | +5.31 | [+3.95, +6.82] | 0.000 |
| offline - live | +9.04 | [+5.55, +12.36] | 0.000 |
| ft_live - live | +0.10 | [-0.43, +0.75] | 0.397 |
| ft_offline - offline | -0.59 | [-1.66, +0.29] | 0.880 |

## Paired bootstrap of ablation steps, valid (58 clips)

| Paired difference | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| e2e_mcbyte_shcmc2_L0 - e2e_rw25_L0 | +0.09 | [-0.44, +0.64] | 0.373 |
| e2e_v3_gated_L0 - e2e_mcbyte_shcmc2_L0 | +1.35 | [+0.80, +1.95] | 0.000 |
| e2e_v3_gated_stitch_L0 - e2e_v3_gated_L0 | +0.87 | [+0.18, +1.61] | 0.003 |
| e2e_v4_link_L0 - e2e_v3_gated_stitch_L0 | -0.00 | [-0.18, +0.15] | 0.524 |
| e2e_v5_pose_L0 - e2e_v4_link_L0 | +0.87 | [+0.49, +1.31] | 0.000 |
| e2e_v5_pose_L0 - e2e_rw25_L0 | +3.17 | [+1.89, +4.54] | 0.000 |
| e2e_mcbyte_shcmc2_Linf - e2e_rw25_Linf | -0.23 | [-1.39, +0.80] | 0.671 |
| e2e_v3_gated_Linf - e2e_mcbyte_shcmc2_Linf | +3.27 | [+2.26, +4.38] | 0.000 |
| e2e_v3_gated_stitch_Linf - e2e_v3_gated_Linf | +0.94 | [-0.84, +2.66] | 0.136 |
| e2e_v4_link_Linf - e2e_v3_gated_stitch_Linf | +0.19 | [+0.02, +0.36] | 0.014 |
| e2e_v5_pose_Linf - e2e_v4_link_Linf | +1.51 | [+0.86, +2.19] | 0.000 |
| e2e_v5_pose_Linf - e2e_rw25_Linf | +5.69 | [+3.23, +8.11] | 0.000 |

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
