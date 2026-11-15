# Frozen-model robustness to synthetic accompaniment paths

MUSDB18-HQ test songs; 30 s scored excerpts and 5 s unscored recurrent-state context. FIR support is read before the context. Original timing and path delays are retained. Accompaniment is re-normalized to -5 dB BVR after each path. No retraining or path tuning.

Protocol `f2c6c349c6fb34972e5c1b36b6cea913a1e7ffc13066057bcb7280183940fbdf`. Frozen paths: dry identity, mild filter + delay, finite synthetic room FIR. Exact coefficients and path parameters: `protocol.json`; checkpoint SHA256 hashes: `run_provenance/`; scored excerpts and dataset-file fingerprints: `excerpts.json`.

Synthetic feed-forward stress tests, not measured rooms, real microphones or acoustic feedback. All comparisons average seeds 0 and 1 per song first. Bootstrap 95% CIs resample songs (10,000 draws, fixed seed 0); they do not include training-seed uncertainty.

## Primary operating point (gamma = 1)

Medians over songs after seed averaging. Vocal loss is output vocal energy relative to clean vocal (0 dB = no energy loss); bleed attenuation is positive when accompaniment energy is removed.

| config | path | songs | SI-SDR (dB) | vocal loss (dB) | bleed attenuation (dB) |
|---|---|---:|---:|---:|---:|
| B_sym256 | dry | 50 | 10.69 | -0.75 | 8.84 |
| B_sym256 | filtered_delay | 50 | 10.73 | -0.74 | 8.83 |
| B_sym256 | synthetic_room | 50 | 10.78 | -0.73 | 8.98 |
| C_dual | dry | 50 | 10.93 | -0.63 | 9.43 |
| C_dual | filtered_delay | 50 | 10.91 | -0.65 | 9.60 |
| C_dual | synthetic_room | 50 | 11.21 | -0.63 | 9.58 |
| D_dual_2p9ms | dry | 50 | 10.59 | -0.74 | 9.29 |
| D_dual_2p9ms | filtered_delay | 50 | 10.73 | -0.76 | 9.36 |
| D_dual_2p9ms | synthetic_room | 50 | 10.97 | -0.74 | 9.31 |
| E_sym128 | dry | 50 | 9.98 | -0.87 | 8.51 |
| E_sym128 | filtered_delay | 50 | 10.06 | -0.84 | 8.49 |
| E_sym128 | synthetic_room | 50 | 10.13 | -0.86 | 8.57 |

## Paired SI-SDR gain at gamma = 1

| comparison | path | paired songs | median difference (dB) [95% CI] | mean difference (dB) [95% CI] |
|---|---|---:|---|---|
| C_dual - B_sym256 | dry | 50 | +0.65 [+0.36, +0.88] | +0.51 [+0.20, +0.78] |
| C_dual - B_sym256 | filtered_delay | 50 | +0.62 [+0.32, +0.91] | +0.49 [+0.18, +0.75] |
| C_dual - B_sym256 | synthetic_room | 50 | +0.59 [+0.30, +0.80] | +0.39 [+0.06, +0.66] |
| D_dual_2p9ms - E_sym128 | dry | 50 | +0.79 [+0.60, +0.92] | +0.64 [+0.32, +0.92] |
| D_dual_2p9ms - E_sym128 | filtered_delay | 50 | +0.78 [+0.56, +0.98] | +0.68 [+0.40, +0.93] |
| D_dual_2p9ms - E_sym128 | synthetic_room | 50 | +0.91 [+0.63, +1.08] | +0.71 [+0.38, +0.98] |

## SI-SDR change from dry at gamma = 1

Negative changes indicate degradation relative to the identity path on the same songs.

| comparison | path | paired songs | median difference (dB) [95% CI] | mean difference (dB) [95% CI] |
|---|---|---:|---|---|
| B_sym256 | filtered_delay | 50 | +0.07 [+0.05, +0.10] | +0.08 [+0.02, +0.13] |
| B_sym256 | synthetic_room | 50 | +0.34 [+0.20, +0.45] | +0.34 [+0.21, +0.48] |
| C_dual | filtered_delay | 50 | +0.03 [-0.02, +0.06] | +0.05 [-0.02, +0.14] |
| C_dual | synthetic_room | 50 | +0.23 [-0.02, +0.35] | +0.21 [+0.08, +0.36] |
| D_dual_2p9ms | filtered_delay | 50 | +0.05 [+0.03, +0.08] | +0.08 [+0.02, +0.14] |
| D_dual_2p9ms | synthetic_room | 50 | +0.26 [+0.10, +0.38] | +0.28 [+0.16, +0.41] |
| E_sym128 | filtered_delay | 50 | +0.06 [+0.00, +0.09] | +0.04 [-0.02, +0.09] |
| E_sym128 | synthetic_room | 50 | +0.21 [+0.11, +0.28] | +0.22 [+0.12, +0.33] |

## Paired vocal preservation at equal attainable attenuation

For each song and model pair, take the midpoint of the attenuation range shared by both configs and all three paths, using seed-averaged five-point gamma curves. Use that same target for dry and transformed paths. Linear interpolation only, no extrapolation; songs without a shared range are excluded. Positive differences mean more vocal energy is kept. This compares energy preservation, not perceptual quality.

| comparison | path | paired songs | median difference (dB) [95% CI] | mean difference (dB) [95% CI] |
|---|---|---:|---|---|
| C_dual - B_sym256 | dry | 49 | +0.25 [+0.17, +0.29] | +0.29 [+0.20, +0.40] |
| C_dual - B_sym256 | filtered_delay | 49 | +0.25 [+0.16, +0.31] | +0.29 [+0.19, +0.39] |
| C_dual - B_sym256 | synthetic_room | 49 | +0.19 [+0.13, +0.28] | +0.22 [+0.14, +0.30] |
| D_dual_2p9ms - E_sym128 | dry | 48 | +0.34 [+0.22, +0.42] | +0.37 [+0.28, +0.47] |
| D_dual_2p9ms - E_sym128 | filtered_delay | 48 | +0.35 [+0.25, +0.44] | +0.39 [+0.31, +0.48] |
| D_dual_2p9ms - E_sym128 | synthetic_room | 48 | +0.32 [+0.27, +0.40] | +0.35 [+0.27, +0.44] |

## Vocal-preservation change from dry at the same attenuation

For each song and model pair, take the midpoint of the attenuation range shared by both configs and all three paths, using seed-averaged five-point gamma curves. Use that same target for dry and transformed paths. Linear interpolation only, no extrapolation; songs without a shared range are excluded. Positive differences mean more vocal energy is kept. This compares energy preservation, not perceptual quality.

| comparison | path | paired songs | median difference (dB) [95% CI] | mean difference (dB) [95% CI] |
|---|---|---:|---|---|
| C_dual | filtered_delay | 49 | +0.00 [-0.00, +0.01] | +0.00 [-0.01, +0.01] |
| C_dual | synthetic_room | 49 | +0.03 [+0.01, +0.09] | +0.06 [+0.03, +0.10] |
| B_sym256 | filtered_delay | 49 | +0.01 [+0.00, +0.01] | +0.00 [-0.01, +0.01] |
| B_sym256 | synthetic_room | 49 | +0.11 [+0.06, +0.14] | +0.13 [+0.07, +0.20] |
| D_dual_2p9ms | filtered_delay | 48 | +0.01 [+0.00, +0.01] | +0.01 [-0.00, +0.03] |
| D_dual_2p9ms | synthetic_room | 48 | +0.07 [+0.02, +0.09] | +0.07 [+0.04, +0.11] |
| E_sym128 | filtered_delay | 48 | +0.00 [-0.01, +0.01] | -0.01 [-0.03, +0.00] |
| E_sym128 | synthetic_room | 48 | +0.07 [+0.04, +0.10] | +0.09 [+0.04, +0.15] |

## Equal-attenuation coverage

| pair | eligible / total songs | median target attenuation (dB) | target range across songs (dB) |
|---|---:|---:|---:|
| C_dual - B_sym256 | 49 / 50 | 9.15 | 6.70 to 14.87 |
| D_dual_2p9ms - E_sym128 | 48 / 50 | 8.85 | 6.47 to 13.88 |

All per-song gamma curves: `per_song.csv`. All paired metrics (including bleed attenuation and vocal-loss changes at gamma = 1): `paired_comparisons.csv`. Equal-attenuation targets and interpolated scores: `equal_attenuation_per_song.csv`. Each per-song cache checks its protocol, checkpoint and excerpt identity; a run is complete only after every song and every path/gamma has been validated and the atomic CSV checksum recorded.
