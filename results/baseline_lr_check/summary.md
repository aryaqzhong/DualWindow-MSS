# Exploratory learning-rate check for the original short-window baselines

Design: [`design.md`](design.md). Post hoc and exploratory. Only the baselines get a second learning rate here, so the comparison is conservative toward the dual window.

## Validation-only selection

| latency | setting | lr | best validation, seed 0 / seed 1 (dB) | mean | selected |
|---|---|---:|---|---:|:---:|
| 5.8 ms | B_sym256 | 0.001 | 3.729 / 3.780 | 3.754 | yes |
| 5.8 ms | B_sym256_lr3e4 | 0.0003 | 3.725 / 3.475 | 3.600 |  |
| 2.9 ms | E_sym128 | 0.001 | 3.368 / 3.392 | 3.380 | yes |
| 2.9 ms | E_sym128_lr3e4 | 0.0003 | 3.282 / 3.216 | 3.249 |  |

## Test songs (50, full length, mono; seeds averaged per song)

| latency | comparison | songs | median difference [95% CI] | mean difference [95% CI] | first better | seed-pair medians |
|---|---|---:|---|---|---:|---|
| 5.8 ms | C_dual - B_sym256 | 50 | +0.76 [+0.57, +0.90] | +0.62 [+0.38, +0.81] | 45 / 50 | +0.64 … +0.89 |
| 2.9 ms | D_dual_2p9ms - E_sym128 | 50 | +0.73 [+0.59, +0.88] | +0.35 [-0.57, +0.91] | 45 / 50 | +0.59 … +0.90 |

| latency | config | SI-SDR median (dB) |
|---|---|---:|
| 5.8 ms | B_sym256 | 3.17 |
| 5.8 ms | C_dual | 4.20 |
| 2.9 ms | E_sym128 | 3.17 |
| 2.9 ms | D_dual_2p9ms | 4.02 |

Bootstrap: 10,000 song resamples, seed 0; intervals exclude training-seed variance. Per-song values: `results/eval/*_test.csv`; table: `paired.csv`.
