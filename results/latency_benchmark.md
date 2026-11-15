# CPU streaming benchmark (full pipeline per hop, 1 thread)

Environment: cpu=Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, threads=1, torch=2.15.0+cu132, onnxruntime=1.31.1, numpy=2.5.4, python=3.12.16, seconds_timed=30.0, warmup_hops=300, peak_rss_mb=556.140625

| config | backend | weights | algorithmic latency (ms) | hop (ms) | P50 / P95 / P99 (µs) | RTF | deadline misses | DSP only (µs) | params | GMAC/s |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| A_sym1024 | torch | runs/A_sym1024_seed0/best.pt | 23.2 | 11.61 | 1289 / 1428 / 1485 | 0.113 | 0.00% | 43 | 939,297 | 0.29 |
| A_sym1024 | onnx | runs/A_sym1024_seed0/best.pt | 23.2 | 11.61 | 563 / 585 / 608 | 0.048 | 0.00% | 43 | 939,297 | 0.29 |
| B_sym256 | torch | runs/B_sym256_seed0/best.pt | 5.8 | 2.90 | 1231 / 1262 / 1279 | 0.427 | 0.09% | 34 | 939,297 | 0.53 |
| B_sym256 | onnx | runs/B_sym256_seed0/best.pt | 5.8 | 2.90 | 393 / 417 / 421 | 0.138 | 0.01% | 34 | 939,297 | 0.53 |
| C_dual | torch | runs/C_dual_seed0/best.pt | 5.8 | 2.90 | 1714 / 1754 / 1789 | 0.592 | 0.03% | 63 | 939,297 | 2.01 |
| C_dual | onnx | runs/C_dual_seed0/best.pt | 5.8 | 2.90 | 750 / 757 / 763 | 0.257 | 0.01% | 63 | 939,297 | 2.01 |
| D_dual_2p9ms | torch | runs/D_dual_2p9ms_seed0/best.pt | 2.9 | 1.45 | 1707 / 1752 / 1790 | 1.180 | 100.00% | 63 | 939,297 | 4.02 |
| D_dual_2p9ms | onnx | runs/D_dual_2p9ms_seed0/best.pt | 2.9 | 1.45 | 750 / 757 / 764 | 0.515 | 0.01% | 63 | 939,297 | 4.02 |
| E_sym128 | torch | runs/E_sym128_seed0/best.pt | 2.9 | 1.45 | 1185 / 1213 / 1220 | 0.820 | 0.00% | 28 | 939,297 | 0.86 |
| E_sym128 | onnx | runs/E_sym128_seed0/best.pt | 2.9 | 1.45 | 372 / 396 / 413 | 0.262 | 0.00% | 28 | 939,297 | 0.86 |
| B_sym256_h64 | torch | runs/B_sym256_h64_seed0/best.pt | 5.8 | 2.90 | 1047 / 1069 / 1077 | 0.362 | 0.01% | 34 | 169,953 | 0.27 |
| B_sym256_h64 | onnx | runs/B_sym256_h64_seed0/best.pt | 5.8 | 2.90 | 243 / 269 / 286 | 0.086 | 0.00% | 34 | 169,953 | 0.27 |
| B_sym256_h128 | torch | runs/B_sym256_h128_seed0/best.pt | 5.8 | 2.90 | 1107 / 1132 / 1138 | 0.382 | 0.00% | 34 | 377,249 | 0.34 |
| B_sym256_h128 | onnx | runs/B_sym256_h128_seed0/best.pt | 5.8 | 2.90 | 291 / 315 / 333 | 0.103 | 0.00% | 34 | 377,249 | 0.34 |
| C_dual_h64 | torch | runs/C_dual_h64_seed0/best.pt | 5.8 | 2.90 | 1519 / 1540 / 1551 | 0.523 | 0.00% | 63 | 169,953 | 1.75 |
| C_dual_h64 | onnx | runs/C_dual_h64_seed0/best.pt | 5.8 | 2.90 | 601 / 608 / 624 | 0.205 | 0.01% | 63 | 169,953 | 1.75 |
| C_dual_h128 | torch | runs/C_dual_h128_seed0/best.pt | 5.8 | 2.90 | 1581 / 1604 / 1623 | 0.544 | 0.00% | 63 | 377,249 | 1.82 |
| C_dual_h128 | onnx | runs/C_dual_h128_seed0/best.pt | 5.8 | 2.90 | 645 / 651 / 669 | 0.220 | 0.00% | 63 | 377,249 | 1.82 |

RTF = mean processing time per hop / hop duration. Algorithmic latency (synthesis window) is separate from processing time; end-to-end latency with an audio device is not measured here.
