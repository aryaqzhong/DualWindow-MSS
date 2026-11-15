# Frozen-model failure analysis

Status: complete.

Original full-song, 50-track results remain the primary evaluation. This is a post-hoc diagnostic of the already inspected test results, not a new independent confirmatory test. No tracks are removed and no models are selected using these diagnostics.

## Protocol

Frozen B/C/D/E best checkpoints, two seeds. Each complete song is inferred continuously; recurrent state is carried across chunks. The same mask predicted from the mixture is separately applied to the vocal and combined accompaniment. Numerical decomposition and agreement with saved full-song SI-SDR/uSDR are checked.

Legacy full-song evaluations used CPU. A numerical audit found that default cuDNN TF32 produced a 0.029 dB discrepancy on C seed 0 / PR - Happy Daze, exceeding the fixed 0.02 dB guard. The failed v1 attempt and its source are retained in `attempt_v1_failed/`. This v2 reruns all tracks with CUDA matmul and cuDNN TF32 disabled, retaining the original 0.02 dB agreement requirement. The audit confirms the decomposition and original GPU inference are sample-identical; full-precision GPU inference reduces the C0 discrepancy to 0.00056 dB. Backend flags are part of run/cache provenance. Audit outputs are in `../failure_analysis_numerics/`.

Activity uses only clean reference vocals in non-overlapping 1 s blocks, including the final short block with its actual sample count: RMS ≥ max(peak-block RMS − 40 dB, −60 dBFS). Fixed sensitivity thresholds are 30 and 50 dB below peak with the same floor, plus −50/−70 dBFS floors at relative 40 dB. Effective thresholds and identical-mask indicators are saved, because the absolute floor can dominate. Selected samples are concatenated only for scoring after continuous inference. Inactive intervals receive output-energy measurements, not SI-SDR. All thresholds are reported; none is chosen by test performance.

## Paired full-song and active-sample results

Seeds are averaged per song before paired song-bootstrap intervals (10,000 draws, seed 0). CIs do not include training-seed uncertainty. Positive vocal-loss differences mean more vocal energy retained; positive accompaniment-attenuation differences mean more accompaniment suppressed.

| pair | metric | dB below peak / floor dBFS | songs | median difference [95% CI] | mean difference [95% CI] | positive songs |
|---|---|---|---:|---|---|---:|
| C − B at 5.8 ms | active_si_sdr | 30 / -60 | 50 | +0.76 [+0.61, +0.90] | +0.66 [+0.50, +0.82] | 46/50 |
| C − B at 5.8 ms | active_usdr | 30 / -60 | 50 | +0.53 [+0.46, +0.60] | +0.49 [+0.38, +0.60] | 47/50 |
| C − B at 5.8 ms | active_vocal_loss_db | 30 / -60 | 50 | +0.40 [+0.33, +0.53] | +0.45 [+0.35, +0.56] | 44/50 |
| C − B at 5.8 ms | active_accompaniment_attenuation_db | 30 / -60 | 50 | +0.29 [+0.04, +0.40] | +0.25 [+0.02, +0.47] | 36/50 |
| C − B at 5.8 ms | whole_si_sdr | — | 50 | +0.76 [+0.57, +0.90] | +0.62 [+0.38, +0.81] | 45/50 |
| C − B at 5.8 ms | whole_usdr | — | 50 | +0.49 [+0.42, +0.63] | +0.45 [+0.28, +0.59] | 46/50 |
| C − B at 5.8 ms | active_si_sdr | 40 / -60 | 50 | +0.76 [+0.61, +0.89] | +0.66 [+0.49, +0.81] | 46/50 |
| C − B at 5.8 ms | active_usdr | 40 / -60 | 50 | +0.52 [+0.46, +0.59] | +0.49 [+0.37, +0.59] | 47/50 |
| C − B at 5.8 ms | active_vocal_loss_db | 40 / -60 | 50 | +0.40 [+0.33, +0.53] | +0.45 [+0.35, +0.56] | 44/50 |
| C − B at 5.8 ms | active_accompaniment_attenuation_db | 40 / -60 | 50 | +0.28 [+0.03, +0.40] | +0.24 [+0.00, +0.46] | 36/50 |
| C − B at 5.8 ms | active_si_sdr | 50 / -60 | 50 | +0.76 [+0.61, +0.88] | +0.66 [+0.49, +0.81] | 46/50 |
| C − B at 5.8 ms | active_usdr | 50 / -60 | 50 | +0.51 [+0.46, +0.59] | +0.49 [+0.37, +0.59] | 47/50 |
| C − B at 5.8 ms | active_vocal_loss_db | 50 / -60 | 50 | +0.40 [+0.33, +0.53] | +0.45 [+0.35, +0.56] | 44/50 |
| C − B at 5.8 ms | active_accompaniment_attenuation_db | 50 / -60 | 50 | +0.28 [+0.04, +0.40] | +0.23 [+0.00, +0.46] | 36/50 |
| C − B at 5.8 ms | active_si_sdr | 40 / -50 | 50 | +0.76 [+0.61, +0.89] | +0.71 [+0.57, +0.84] | 47/50 |
| C − B at 5.8 ms | active_usdr | 40 / -50 | 50 | +0.53 [+0.46, +0.59] | +0.50 [+0.39, +0.60] | 47/50 |
| C − B at 5.8 ms | active_vocal_loss_db | 40 / -50 | 50 | +0.40 [+0.32, +0.53] | +0.45 [+0.35, +0.56] | 44/50 |
| C − B at 5.8 ms | active_accompaniment_attenuation_db | 40 / -50 | 50 | +0.29 [+0.09, +0.40] | +0.30 [+0.06, +0.52] | 37/50 |
| C − B at 5.8 ms | active_si_sdr | 40 / -70 | 50 | +0.75 [+0.61, +0.89] | +0.65 [+0.48, +0.81] | 46/50 |
| C − B at 5.8 ms | active_usdr | 40 / -70 | 50 | +0.52 [+0.45, +0.59] | +0.48 [+0.36, +0.59] | 47/50 |
| C − B at 5.8 ms | active_vocal_loss_db | 40 / -70 | 50 | +0.40 [+0.33, +0.53] | +0.45 [+0.35, +0.56] | 44/50 |
| C − B at 5.8 ms | active_accompaniment_attenuation_db | 40 / -70 | 50 | +0.28 [+0.03, +0.40] | +0.23 [-0.00, +0.45] | 36/50 |
| D − E at 2.9 ms | active_si_sdr | 30 / -60 | 50 | +0.77 [+0.62, +0.89] | +0.57 [+0.06, +0.93] | 45/50 |
| D − E at 2.9 ms | active_usdr | 30 / -60 | 50 | +0.56 [+0.43, +0.64] | +0.60 [+0.46, +0.74] | 46/50 |
| D − E at 2.9 ms | active_vocal_loss_db | 30 / -60 | 50 | +0.36 [+0.28, +0.53] | +0.21 [-0.48, +0.64] | 44/50 |
| D − E at 2.9 ms | active_accompaniment_attenuation_db | 30 / -60 | 50 | +0.60 [+0.38, +0.97] | +0.72 [+0.16, +1.45] | 37/50 |
| D − E at 2.9 ms | whole_si_sdr | — | 50 | +0.73 [+0.59, +0.88] | +0.35 [-0.57, +0.91] | 45/50 |
| D − E at 2.9 ms | whole_usdr | — | 50 | +0.55 [+0.43, +0.63] | +0.53 [+0.39, +0.66] | 44/50 |
| D − E at 2.9 ms | active_si_sdr | 40 / -60 | 50 | +0.76 [+0.60, +0.88] | +0.57 [+0.05, +0.92] | 45/50 |
| D − E at 2.9 ms | active_usdr | 40 / -60 | 50 | +0.55 [+0.43, +0.63] | +0.59 [+0.46, +0.74] | 46/50 |
| D − E at 2.9 ms | active_vocal_loss_db | 40 / -60 | 50 | +0.36 [+0.28, +0.53] | +0.21 [-0.48, +0.64] | 44/50 |
| D − E at 2.9 ms | active_accompaniment_attenuation_db | 40 / -60 | 50 | +0.60 [+0.38, +0.95] | +0.71 [+0.16, +1.45] | 37/50 |
| D − E at 2.9 ms | active_si_sdr | 50 / -60 | 50 | +0.76 [+0.60, +0.88] | +0.56 [+0.05, +0.92] | 45/50 |
| D − E at 2.9 ms | active_usdr | 50 / -60 | 50 | +0.55 [+0.43, +0.63] | +0.59 [+0.46, +0.74] | 46/50 |
| D − E at 2.9 ms | active_vocal_loss_db | 50 / -60 | 50 | +0.36 [+0.28, +0.53] | +0.21 [-0.48, +0.64] | 44/50 |
| D − E at 2.9 ms | active_accompaniment_attenuation_db | 50 / -60 | 50 | +0.60 [+0.38, +0.94] | +0.71 [+0.16, +1.45] | 37/50 |
| D − E at 2.9 ms | active_si_sdr | 40 / -50 | 50 | +0.77 [+0.61, +0.89] | +0.58 [+0.09, +0.92] | 45/50 |
| D − E at 2.9 ms | active_usdr | 40 / -50 | 50 | +0.56 [+0.43, +0.64] | +0.59 [+0.46, +0.73] | 46/50 |
| D − E at 2.9 ms | active_vocal_loss_db | 40 / -50 | 50 | +0.36 [+0.28, +0.53] | +0.21 [-0.48, +0.64] | 44/50 |
| D − E at 2.9 ms | active_accompaniment_attenuation_db | 40 / -50 | 50 | +0.60 [+0.38, +0.96] | +0.73 [+0.15, +1.51] | 37/50 |
| D − E at 2.9 ms | active_si_sdr | 40 / -70 | 50 | +0.76 [+0.60, +0.88] | +0.57 [+0.06, +0.92] | 45/50 |
| D − E at 2.9 ms | active_usdr | 40 / -70 | 50 | +0.56 [+0.43, +0.63] | +0.60 [+0.46, +0.74] | 46/50 |
| D − E at 2.9 ms | active_vocal_loss_db | 40 / -70 | 50 | +0.36 [+0.28, +0.53] | +0.21 [-0.48, +0.64] | 44/50 |
| D − E at 2.9 ms | active_accompaniment_attenuation_db | 40 / -70 | 50 | +0.58 [+0.38, +0.95] | +0.71 [+0.16, +1.46] | 37/50 |

## Selected cases

Case selection was fixed from saved full-song differences before computing activity diagnostics: five lowest gains and three positive-gain rank quantiles for each latency pair. This deliberately includes failures; these cases are not representative estimates of all-song quality.

| pair | selection | track | full gain | active gain (primary) | active samples / vocal energy kept | threshold | vocal RMS | vocal/acc energy |
|---|---|---|---:|---:|---|---:|---:|---:|
| C − B at 5.8 ms | worst_1 | PR - Oh No | -2.73 | -1.37 | 45.9% / 100.0% | -60.0 dBFS | -36.2 dBFS | -19.6 dB |
| C − B at 5.8 ms | worst_2 | PR - Happy Daze | -2.25 | -1.47 | 32.6% / 100.0% | -60.0 dBFS | -41.1 dBFS | -26.1 dB |
| C − B at 5.8 ms | worst_3 | Detsky Sad - Walkie Talkie | -0.29 | +0.33 | 33.6% / 100.0% | -58.8 dBFS | -27.2 dBFS | -7.9 dB |
| C − B at 5.8 ms | worst_4 | The Easton Ellises - Falcon 69 | -0.20 | -0.08 | 81.8% / 100.0% | -58.6 dBFS | -23.8 dBFS | -5.9 dB |
| C − B at 5.8 ms | worst_5 | Buitraker - Revo X | -0.10 | -0.09 | 71.8% / 100.0% | -60.0 dBFS | -28.5 dBFS | -9.6 dB |
| C − B at 5.8 ms | positive_rank_quantile_0.25 | Angels In Amplifiers - I'm Alright | +0.55 | +0.85 | 81.3% / 100.0% | -59.0 dBFS | -24.1 dBFS | -3.1 dB |
| C − B at 5.8 ms | positive_rank_quantile_0.5 | BKS - Bulldozer | +0.81 | +1.15 | 61.1% / 100.0% | -60.0 dBFS | -33.3 dBFS | -13.7 dB |
| C − B at 5.8 ms | positive_rank_quantile_0.75 | BKS - Too Much | +1.07 | +1.05 | 62.7% / 100.0% | -59.6 dBFS | -25.8 dBFS | -4.7 dB |
| D − E at 2.9 ms | worst_1 | PR - Happy Daze | -20.06 | -10.15 | 32.6% / 100.0% | -60.0 dBFS | -41.1 dBFS | -26.1 dB |
| D − E at 2.9 ms | worst_2 | Detsky Sad - Walkie Talkie | -1.18 | +0.29 | 33.6% / 100.0% | -58.8 dBFS | -27.2 dBFS | -7.9 dB |
| D − E at 2.9 ms | worst_3 | PR - Oh No | -1.14 | -1.52 | 45.9% / 100.0% | -60.0 dBFS | -36.2 dBFS | -19.6 dB |
| D − E at 2.9 ms | worst_4 | Moosmusic - Big Dummy Shake | -0.41 | -0.17 | 76.9% / 100.0% | -56.6 dBFS | -23.2 dBFS | -5.3 dB |
| D − E at 2.9 ms | worst_5 | Buitraker - Revo X | -0.04 | -0.04 | 71.8% / 100.0% | -60.0 dBFS | -28.5 dBFS | -9.6 dB |
| D − E at 2.9 ms | positive_rank_quantile_0.25 | Mu - Too Bright | +0.58 | +0.58 | 79.0% / 100.0% | -60.0 dBFS | -25.9 dBFS | -5.2 dB |
| D − E at 2.9 ms | positive_rank_quantile_0.5 | M.E.R.C. Music - Knockout | +0.81 | +0.81 | 86.3% / 100.0% | -59.5 dBFS | -25.8 dBFS | -3.1 dB |
| D − E at 2.9 ms | positive_rank_quantile_0.75 | Speak Softly - Like Horses | +1.06 | +1.11 | 83.2% / 100.0% | -55.6 dBFS | -22.5 dBFS | -3.4 dB |

## Extreme-case details: PR - Happy Daze

| model | full SI-SDR | active SI-SDR | full uSDR | active uSDR | full cosine | active cosine | active vocal loss | active acc attenuation | inactive output | inactive output-energy share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_sym256 | -25.87 | -24.03 | -0.92 | -0.56 | 0.07869 | 0.09421 | -20.46 dB | 31.68 dB | -51.7 dBFS | 34.1% |
| C_dual | -28.12 | -25.50 | -0.89 | -0.43 | 0.04069 | 0.05426 | -20.14 dB | 31.27 dB | -49.3 dBFS | 44.9% |
| D_dual_2p9ms | -40.13 | -29.32 | -2.24 | -0.11 | 0.01684 | 0.03419 | -25.88 dB | 37.44 dB | -47.3 dBFS | 59.4% |
| E_sym128 | -20.07 | -19.17 | -3.18 | -2.62 | 0.10022 | 0.11161 | -11.10 dB | 22.35 dB | -46.1 dBFS | 18.4% |

Activity sensitivity for this track (reference-only, identical across models):

| relative dB | floor dBFS | effective threshold dBFS | active samples | reference energy kept | same mask as primary |
|---:|---:|---:|---:|---:|---|
| 30 | -60 | -60.0 | 32.6% | 99.96% | yes |
| 40 | -70 | -70.0 | 36.3% | 99.99% | no |
| 40 | -60 | -60.0 | 32.6% | 99.96% | yes |
| 40 | -50 | -50.0 | 26.4% | 99.75% | no |
| 50 | -60 | -60.0 | 32.6% | 99.96% | yes |

## Interpretation limits

A low-energy or sparse reference can make full-song projection metrics sensitive to residual output, but it is not evidence that a bad score is merely an artifact. Check active SI-SDR, uSDR, the uncentered cosine/projection coefficient, vocal preservation and residual accompaniment together. A score change after masking activity changes the evaluation question and does not replace the full-song result. Preserved-vocal energy alone is not perceptual naturalness; no listener study was performed.

`per_track.csv` contains seed-averaged metrics; the eight run CSVs retain individual seeds. `cases.csv` contains case-level details. Block-level numeric curves are in `blocks/`; no MUSDB audio is exported. Source files are identified by path/size/mtime, not content hashes. Checkpoints, code and output CSVs have SHA-256 provenance, and per-song caches validate identities and payload hashes.
