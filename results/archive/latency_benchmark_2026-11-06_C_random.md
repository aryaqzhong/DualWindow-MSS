# CPU streaming benchmark (full pipeline per hop, 1 thread)

Environment: cpu=Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, threads=1, torch=2.15.0+cu132, onnxruntime=1.31.1, numpy=2.5.4, python=3.12.16, seconds_timed=30.0, warmup_hops=300, peak_rss_mb=548.22265625

| config | backend | weights | algorithmic latency (ms) | hop (ms) | P50 / P95 / P99 (µs) | RTF | deadline misses | DSP only (µs) | params | GMAC/s |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| A_sym1024 | torch | runs/A_sym1024_seed0/best.pt | 23.2 | 11.61 | 1689 / 1949 / 2032 | 0.142 | 0.00% | 45 | 939,297 | 0.29 |
| A_sym1024 | onnx | runs/A_sym1024_seed0/best.pt | 23.2 | 11.61 | 610 / 843 / 907 | 0.056 | 0.00% | 45 | 939,297 | 0.29 |
| B_sym256 | torch | runs/B_sym256_seed0/best.pt | 5.8 | 2.90 | 1408 / 1720 / 1823 | 0.481 | 0.07% | 30 | 939,297 | 0.53 |
| B_sym256 | onnx | runs/B_sym256_seed0/best.pt | 5.8 | 2.90 | 397 / 611 / 694 | 0.149 | 0.00% | 30 | 939,297 | 0.53 |
| C_dual | torch | random | 5.8 | 2.90 | 2028 / 2322 / 2415 | 0.687 | 0.02% | 59 | 939,297 | 2.01 |
| C_dual | onnx | random | 5.8 | 2.90 | 845 / 1099 / 1175 | 0.300 | 0.01% | 59 | 939,297 | 2.01 |
| D_dual_2p9ms | torch | random | 2.9 | 1.45 | 2043 / 2355 / 2454 | 1.376 | 100.00% | 59 | 939,297 | 4.02 |
| D_dual_2p9ms | onnx | random | 2.9 | 1.45 | 851 / 1087 / 1155 | 0.599 | 0.00% | 59 | 939,297 | 4.02 |

RTF = mean processing time per hop / hop duration. Algorithmic latency (synthesis window) is separate from processing time; end-to-end latency with an audio device is not measured here.
