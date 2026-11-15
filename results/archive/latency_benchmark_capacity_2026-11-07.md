# CPU streaming benchmark (full pipeline per hop, 1 thread)

Environment: cpu=Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, threads=1, torch=2.15.0+cu132, onnxruntime=1.31.1, numpy=2.5.4, python=3.12.16, seconds_timed=30.0, warmup_hops=300, peak_rss_mb=543.25

| config | backend | weights | algorithmic latency (ms) | hop (ms) | P50 / P95 / P99 (µs) | RTF | deadline misses | DSP only (µs) | params | GMAC/s |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| B_sym256_h64 | torch | runs/B_sym256_h64_seed0/best.pt | 5.8 | 2.90 | 957 / 1133 / 1215 | 0.338 | 0.05% | 28 | 169,953 | 0.27 |
| B_sym256_h64 | onnx | runs/B_sym256_h64_seed0/best.pt | 5.8 | 2.90 | 238 / 278 / 306 | 0.087 | 0.05% | 28 | 169,953 | 0.27 |
| B_sym256_h128 | torch | runs/B_sym256_h128_seed0/best.pt | 5.8 | 2.90 | 1103 / 1347 / 1439 | 0.389 | 0.02% | 28 | 377,249 | 0.34 |
| B_sym256_h128 | onnx | runs/B_sym256_h128_seed0/best.pt | 5.8 | 2.90 | 300 / 341 / 361 | 0.106 | 0.00% | 28 | 377,249 | 0.34 |
| B_sym256 | torch | runs/B_sym256_seed0/best.pt | 5.8 | 2.90 | 1370 / 1691 / 1777 | 0.478 | 0.02% | 28 | 939,297 | 0.53 |
| B_sym256 | onnx | runs/B_sym256_seed0/best.pt | 5.8 | 2.90 | 446 / 522 / 594 | 0.156 | 0.00% | 28 | 939,297 | 0.53 |
| C_dual_h64 | torch | runs/C_dual_h64_seed0/best.pt | 5.8 | 2.90 | 1623 / 1890 / 1976 | 0.562 | 0.02% | 59 | 169,953 | 1.75 |
| C_dual_h64 | onnx | runs/C_dual_h64_seed0/best.pt | 5.8 | 2.90 | 614 / 665 / 703 | 0.213 | 0.02% | 59 | 169,953 | 1.75 |
| C_dual_h128 | torch | runs/C_dual_h128_seed0/best.pt | 5.8 | 2.90 | 1725 / 1997 / 2081 | 0.595 | 0.01% | 59 | 377,249 | 1.82 |
| C_dual_h128 | onnx | runs/C_dual_h128_seed0/best.pt | 5.8 | 2.90 | 655 / 725 / 772 | 0.228 | 0.00% | 59 | 377,249 | 1.82 |
| C_dual | torch | runs/C_dual_seed0/best.pt | 5.8 | 2.90 | 1970 / 2265 / 2369 | 0.675 | 0.02% | 60 | 939,297 | 2.01 |
| C_dual | onnx | runs/C_dual_seed0/best.pt | 5.8 | 2.90 | 804 / 981 / 1080 | 0.284 | 0.00% | 60 | 939,297 | 2.01 |

RTF = mean processing time per hop / hop duration. Algorithmic latency (synthesis window) is separate from processing time; end-to-end latency with an audio device is not measured here.
