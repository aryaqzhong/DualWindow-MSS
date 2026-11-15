# Exploratory learning-rate check for the 5.8 ms wide short-window control

Design: [`design.md`](design.md). Post hoc and exploratory; `C_dual` gets no extra learning-rate candidate, so the comparison is conservative toward the dual window.

## Validation-only selection

| setting | lr | best validation, seed 0 / seed 1 (dB) | mean |
|---|---:|---|---:|
| B_sym256_compute | 0.001 | 3.101 / 3.112 | 3.107 |
| B_sym256_compute_lr3e4 | 0.0003 | 3.860 / 3.800 | 3.830 |

Selected: **B_sym256_compute_lr3e4**.

## Test songs (50, full length, mono; seeds averaged per song)

| config | SI-SDR median (dB) |
|---|---:|
| B_sym256 | 3.17 |
| B_sym256_compute_lr3e4 | 3.63 |
| C_dual | 4.20 |

| comparison | songs | median difference [95% CI] | first better | seed-pair medians |
|---|---:|---|---:|---|
| C_dual - B_sym256_compute_lr3e4 | 50 | +0.51 [+0.40, +0.65] | 42 / 50 | +0.36 … +0.68 |
| B_sym256_compute_lr3e4 - B_sym256 | 50 | +0.23 [+0.15, +0.32] | 42 / 50 | +0.04 … +0.39 |
| C_dual - B_sym256_compute | 50 | +1.39 [+1.22, +1.63] | 49 / 50 | +1.31 … +1.54 |
| B_sym256_compute_lr3e4 - B_sym256_compute | 50 | +0.82 [+0.68, +1.04] | 48 / 50 | +0.71 … +0.94 |

Bootstrap: 10,000 song resamples, seed 0; intervals exclude training-seed variance. Per-song values: `results/eval/*_test.csv`; table: `paired.csv`.
