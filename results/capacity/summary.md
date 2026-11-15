# Capacity experiment: dual-window gain vs. model size

B = 256/256 and C = 2048/256 (both 5.8 ms algorithmic latency); only the GRU width changes. MUSDB18-HQ test set (full songs, mono, vocals). SI-SDR in dB, median over songs of the seed-averaged value. A (1024/1024, 23.2 ms, GRU 256) is listed as the long-window reference.

| config | GRU width | params | MMAC / frame | GMAC / s | seeds | SI-SDR per seed | SI-SDR | SI-SDRi | RTF (1 thread) | P99 per hop (µs) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| A_sym1024 | 256 | 939,297 | 3.39 | 0.29 | 2 | 4.64 / 4.68 | 4.65 | 10.57 | 0.048 | 608 |
| B_sym256_h64 | 64 | 169,953 | 0.78 | 0.27 | 2 | 3.33 / 3.08 | 3.12 | 9.36 | 0.086 | 286 |
| B_sym256_h128 | 128 | 377,249 | 0.99 | 0.34 | 2 | 2.99 / 3.15 | 3.06 | 9.32 | 0.103 | 333 |
| B_sym256 | 256 | 939,297 | 1.55 | 0.53 | 2 | 3.30 / 3.15 | 3.17 | 9.46 | 0.138 | 421 |
| C_dual_h64 | 64 | 169,953 | 5.07 | 1.75 | 2 | 3.82 / 3.93 | 3.82 | 9.93 | 0.205 | 624 |
| C_dual_h128 | 128 | 377,249 | 5.28 | 1.82 | 2 | 4.07 / 4.11 | 4.10 | 10.04 | 0.220 | 669 |
| C_dual | 256 | 939,297 | 5.84 | 2.01 | 2 | 4.27 / 4.16 | 4.20 | 10.15 | 0.257 | 763 |

## Dual-window gain C − B per width (paired per song, seed-averaged)

| GRU width | params | songs | median gain [95% CI] | mean gain [95% CI] | songs where C is better | median gain over seed pairings | gain minus gain at width 256: median [95% CI] / mean [95% CI] |
|---:|---:|---:|---|---|---:|---|---|
| 64 | 169,953 | 50 | +0.54 [+0.40, +0.69] | +0.46 [+0.27, +0.62] | 80% | +0.57 … +0.59 (4) | -0.10 [-0.23, -0.01] / -0.16 [-0.35, +0.05] |
| 128 | 377,249 | 50 | +0.59 [+0.50, +0.71] | +0.32 [-0.16, +0.65] | 84% | +0.54 … +0.70 (4) | -0.08 [-0.20, +0.01] / -0.30 [-0.65, -0.04] |
| 256 | 939,297 | 50 | +0.76 [+0.57, +0.90] | +0.62 [+0.38, +0.81] | 90% | +0.64 … +0.89 (4) | – |

## Effect of model size (paired per song, seed-averaged)

| comparison | params | MMAC / frame | median difference [95% CI] | mean difference [95% CI] | songs where the first is better |
|---|---|---|---|---|---:|
| B: GRU 256 − GRU 64 | 0.94 M vs. 0.17 M | 1.55 vs. 0.78 | +0.32 [+0.19, +0.39] | +0.26 [+0.10, +0.44] | 80% |
| C: GRU 256 − GRU 64 | 0.94 M vs. 0.17 M | 5.84 vs. 5.07 | +0.39 [+0.32, +0.51] | +0.42 [+0.25, +0.59] | 90% |
| C at GRU 64 − B at GRU 256 | 0.17 M vs. 0.94 M | 5.07 vs. 1.55 | +0.31 [+0.19, +0.46] | +0.19 [+0.00, +0.36] | 72% |

Bootstrap CIs resample songs (10,000 resamples) and do not include training-seed variance; the seed-pairing column shows that variance separately (median gain for each of the 2 × 2 pairings of seeds). A positive last column means the dual window helps the smaller model more than the width-256 model.

The median is the primary summary. A few songs have almost no vocal: on PR - Happy Daze (mixture SI-SDR -25.4 dB, test-set median -5.8 dB) the single B/C runs score -17 to -56 dB, so one run can move the mean over 50 songs by several tenths of a dB.

Compute note: the frequency-direction encoder and the per-bin head run on every FFT bin, so C (1025 bins) costs more per frame than B (129 bins) at the same width; the GRU width mainly changes the parameter count. MACs are counted from the layer shapes (dwmss/complexity.py).
