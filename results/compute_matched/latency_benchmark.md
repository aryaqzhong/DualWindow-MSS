# CPU streaming benchmark (full pipeline per hop, 1 thread)

Environment: cpu=Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, threads=1, torch=2.15.0+cu132, onnxruntime=1.31.1, numpy=2.5.4, python=3.12.16, seconds_timed=30.0, warmup_hops=300, peak_rss_mb=652.7109375

| config | backend | weights | algorithmic latency (ms) | hop (ms) | P50 / P95 / P99 (µs) | RTF | deadline misses | DSP only (µs) | params | GMAC/s |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| B_sym256 | onnx | runs/B_sym256_seed0/best.pt | 5.8 | 2.90 | 382 / 405 / 410 | 0.135 | 0.02% | 34 | 939,297 | 0.53 |
| B_sym256_compute | onnx | runs/B_sym256_compute_seed0/best.pt | 5.8 | 2.90 | 1421 / 1537 / 1581 | 0.479 | 0.03% | 34 | 5,244,377 | 2.02 |
| C_dual | onnx | runs/C_dual_seed0/best.pt | 5.8 | 2.90 | 741 / 746 / 750 | 0.254 | 0.01% | 62 | 939,297 | 2.01 |
| E_sym128 | onnx | runs/E_sym128_seed0/best.pt | 2.9 | 1.45 | 359 / 382 / 384 | 0.253 | 0.00% | 28 | 939,297 | 0.86 |
| E_sym128_compute | onnx | runs/E_sym128_compute_seed0/best.pt | 2.9 | 1.45 | 1476 / 1602 / 1631 | 1.003 | 52.68% | 32 | 5,521,409 | 4.01 |
| D_dual_2p9ms | onnx | runs/D_dual_2p9ms_seed0/best.pt | 2.9 | 1.45 | 737 / 743 / 750 | 0.505 | 0.02% | 63 | 939,297 | 4.02 |

RTF = mean processing time per hop / hop duration. Algorithmic latency (synthesis window) is separate from processing time; end-to-end latency with an audio device is not measured here.
