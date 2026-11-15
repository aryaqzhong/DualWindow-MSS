# Test-set results (MUSDB18-HQ, 50 songs, full length, mono, vocals)

SI-SDR in dB, median over songs. Oracle = IRM2 with the same window configuration (non-deployable reference).

| config | latency (ms) | seeds | SI-SDR per seed | SI-SDR (seed mean) | SI-SDRi (seed mean) | oracle IRM2 SI-SDR |
|---|---:|---:|---|---:|---:|---:|
| A_sym1024 | 23.2 | 2 | 4.64 / 4.68 | 4.65 | 10.57 | 9.95 |
| B_sym256 | 5.8 | 2 | 3.30 / 3.15 | 3.17 | 9.46 | 7.30 |
| C_dual | 5.8 | 2 | 4.27 / 4.16 | 4.20 | 10.15 | 9.25 |
| D_dual_2p9ms | 2.9 | 2 | 3.99 / 4.01 | 4.02 | 9.92 | 8.91 |
| E_sym128 | 2.9 | 2 | 3.08 / 3.26 | 3.17 | 9.07 | 6.71 |

## Paired comparisons (per song, seed-averaged SI-SDR)

| comparison | songs | median difference [95% CI] | mean difference [95% CI] | songs where first is better |
|---|---:|---|---|---:|
| C − B (dual vs. short window, both 5.8 ms) | 50 | +0.76 [+0.57, +0.90] | +0.62 [+0.38, +0.81] | 90% |
| A − C (23.2 ms vs. 5.8 ms dual) | 50 | +0.31 [+0.22, +0.45] | +0.43 [+0.15, +0.86] | 74% |
| A − B (23.2 ms vs. 5.8 ms short) | 50 | +1.05 [+0.87, +1.31] | +1.05 [+0.73, +1.40] | 94% |
| D − E (dual vs. short window, both 2.9 ms) | 50 | +0.73 [+0.59, +0.88] | +0.35 [-0.57, +0.91] | 90% |
| C − D (5.8 ms vs. 2.9 ms dual) | 50 | +0.16 [+0.11, +0.24] | +0.34 [-0.03, +0.90] | 78% |
| B − E (5.8 ms vs. 2.9 ms short) | 50 | +0.13 [+0.03, +0.33] | +0.07 [-0.25, +0.33] | 66% |

Bootstrap CIs resample songs (10,000 resamples). The median is the primary summary because a few songs have almost no vocal: on PR - Happy Daze (mixture SI-SDR -25.4 dB, test-set median -5.8 dB) the single runs score -13 to -50 dB, and such songs dominate the mean.

## Offline reference

HTDemucs (pretrained, non-causal, ~7.8 s segments; same mono input duplicated to stereo): SI-SDR 8.49 dB, SI-SDRi 15.04 dB (medians over 50 songs). Per song it is +4.15 dB [+3.75, +4.58] above A (23.2 ms). It has no low-latency operating point and was trained on more data (`scripts/htdemucs_reference.py`).
