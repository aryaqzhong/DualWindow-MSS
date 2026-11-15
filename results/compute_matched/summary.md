# MAC-matched short-window controls

Status: complete.

Design fixed before training: [`design.md`](design.md), full grid and resolved configurations in `design.json`. Only the short-window GRU width and its projections change. This compares how a fixed model-MAC budget is allocated; it does not hold parameter count or CPU runtime constant.

## Quality and measured cost

SI-SDR: median over all 50 full-length mono test songs after averaging seeds 0 and 1 per song. Per-seed entries are separate song medians, whose mean need not equal the seed-averaged song median.

| config | hidden | params | MMAC/frame | GMAC/s | seeds | SI-SDR seed0 / seed1 | SI-SDR | ONNX RTF | P99 ms (hop ms) | late hops |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| B_sym256 | 256 | 939,297 | 1.552 | 0.535 | 2 | 3.30 / 3.15 | 3.17 | 0.135 | 0.410 (2.902) | 0.02% |
| B_sym256_compute | 776 | 5,244,377 | 5.853 | 2.017 | 2 | 2.61 / 2.83 | 2.72 | 0.479 | 1.581 (2.902) | 0.03% |
| C_dual | 256 | 939,297 | 5.838 | 2.011 | 2 | 4.27 / 4.16 | 4.20 | 0.254 | 0.750 (2.902) | 0.01% |
| E_sym128 | 256 | 939,297 | 1.245 | 0.858 | 2 | 3.08 / 3.26 | 3.17 | 0.253 | 0.384 (1.451) | 0.00% |
| E_sym128_compute | 800 | 5,521,409 | 5.824 | 4.013 | 2 | 2.29 / 2.60 | 2.51 | 1.003 | 1.631 (1.451) | 52.68% |
| D_dual_2p9ms | 256 | 939,297 | 5.838 | 4.023 | 2 | 3.99 / 4.01 | 4.02 | 0.505 | 0.750 (1.451) | 0.02% |

## Paired comparisons

Positive differences favor the first model. The primary comparisons test whether the dual window retains a gain after allocating comparable model MACs to a larger short-window network.

| role | first minus second | songs | median difference [95% CI] | mean difference [95% CI] | first better | seed-pair median range |
|---|---|---:|---|---|---:|---|
| primary | C_dual − B_sym256_compute | 50 | +1.39 [+1.22, +1.63] | +1.32 [+0.94, +1.61] | 49 / 50 | +1.31 … +1.54 |
| primary | D_dual_2p9ms − E_sym128_compute | 50 | +1.56 [+1.19, +1.65] | +1.03 [+0.01, +1.63] | 48 / 50 | +1.34 … +1.57 |
| secondary | B_sym256_compute − B_sym256 | 50 | -0.64 [-0.83, -0.47] | -0.70 [-1.01, -0.40] | 3 / 50 | -0.71 … -0.53 |
| secondary | E_sym128_compute − E_sym128 | 50 | -0.67 [-0.82, -0.54] | -0.68 [-0.93, -0.44] | 8 / 50 | -0.87 … -0.53 |
| reference | C_dual − B_sym256 | 50 | +0.76 [+0.57, +0.90] | +0.62 [+0.38, +0.81] | 45 / 50 | +0.64 … +0.89 |
| reference | D_dual_2p9ms − E_sym128 | 50 | +0.73 [+0.59, +0.88] | +0.35 [-0.57, +0.91] | 45 / 50 | +0.59 … +0.90 |

Intervals resample songs 10,000 times with seed 0. They exclude training-seed variation; the last column gives all four seed-pair medians separately. Per-song seed-averaged values are in `per_track.csv`; all comparison statistics are in `paired.csv`.

## CPU measurement and scope

One job, one CPU thread, ONNX Runtime and trained seed-0 checkpoints for all six models. The full per-hop STFT–network–iSTFT pipeline is timed. Environment: cpu=Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, threads=1, onnxruntime=1.31.1, seconds_timed=30.0.

MAC matching is prespecified; CPU time is reported afterwards and does not select the controls. The wider controls have more parameters and can require different training wall-clock time despite the same 200,000-update budget. Results test this resource allocation under the shared training recipe; they do not establish that either architecture is optimal at a given CPU budget. Algorithmic latency is determined by the synthesis window. Audio-device end-to-end latency is not measured. Each new control's evaluation is associated with SHA-256 hashes of its checkpoint and result files; the report rejects artifacts changed after recording.
