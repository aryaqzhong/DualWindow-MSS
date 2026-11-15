# Failure case: PR - Happy Daze

This post-hoc case study examines the song with the lowest original D-minus-E full-song SI-SDR difference.
It covers one complete 162.70-second test song and all four original D/E validation-best checkpoints
(two seeds per model). It is a case diagnosis, not a population-level performance estimate. The primary
[50-song results](../summary_test.md) retain this song.

The completed full-precision analysis covers all 50 songs and both seeds of B/C/D/E. This case uses its
[full diagnostic tables](per_track.csv), [report](summary.md), and [checkpoint/source/output provenance](summary_provenance.json).
CUDA matmul and cuDNN TF32 were disabled after a separate [numerical audit](../failure_analysis_numerics/summary.md);
maximum SI-SDR disagreement with the original CPU evaluation across all runs is 0.000557 dB. The earlier
single-song smoke and failed initial all-song attempt remain archived. The full diagnostic protocol is in
[EXPERIMENTS.md](../../EXPERIMENTS.md#failure-analysis).

## What makes this case difficult?

The clean vocal RMS is -41.08 dBFS, with vocal energy 26.06 dB below the accompaniment. Reference-only
activity at max(peak one-second vocal RMS minus 40 dB, -60 dBFS) selects 32.6% of samples while retaining
99.96% of the reference vocal energy. Inference still processes the entire continuous song; sample
selection changes scoring only. The activity mask uses the clean reference and is an offline diagnostic.

## Suppression and vocal loss

Values below average each seed's metric, not the two output waveforms. SI-SDR and uSDR use the same
definitions as the primary evaluation. Energy decomposition applies the unchanged mixture-predicted mask
to the clean vocal and accompaniment separately; their outputs sum to the actual estimate numerically.

| metric | D dual window | E short window |
|---|---:|---:|
| Full-song SI-SDR (dB) | -40.13 | -20.07 |
| Vocal-active SI-SDR (dB) | -29.32 | -19.17 |
| Full-song uSDR (dB) | -2.24 | -3.18 |
| Vocal-active uSDR (dB) | -0.11 | -2.62 |
| Active preserved-vocal energy relative to clean vocal (dB) | -25.88 | -11.10 |
| Active accompaniment attenuation (dB, positive = more suppression) | 37.44 | 22.35 |
| Active preserved-vocal-component SI-SDR (dB) | -19.08 | -8.12 |
| Inactive intervals’ share of total output energy | 59.4% | 18.4% |
| Active output/reference uncentered cosine, seed mean | 0.03419 | 0.11161 |

The full-song D-minus-E SI-SDR deficit is about 20.06 dB. Restricting scoring to reference-active samples
reduces it to 10.15 dB but does not remove the failure. D suppresses about 15.1 dB more accompaniment and
retains about 14.8 dB less vocal energy. The preserved-vocal component itself has SI-SDR -19.08 dB for D versus -8.12 dB for E, indicating shape
distortion in addition to attenuation. This is evidence of a severe suppression/preservation trade-off
in this case, alongside a contribution from inactive-interval output to the full-song score. It does not
establish the training or architectural cause.

uSDR ranks D above E here. That does not establish better vocal fidelity: this waveform-error metric is
gain-sensitive, and an all-zero output has approximately 0 dB uSDR against a nonzero reference. D's active
uSDR close to zero therefore has to be read together with the large vocal-energy loss and low correlation.
Low target level by itself is not a reason to dismiss an SI-SDR failure as a metric artifact.

## Fixed threshold sensitivity

| relative threshold below peak / absolute floor | active samples | vocal energy retained by activity mask | active D-minus-E SI-SDR (dB) |
|---|---:|---:|---:|
| 30 dB / -60 dBFS | 32.6% | 99.96% | -10.15 |
| 40 dB / -60 dBFS (primary) | 32.6% | 99.96% | -10.15 |
| 50 dB / -60 dBFS | 32.6% | 99.96% | -10.15 |
| 40 dB / -50 dBFS | 26.4% | 99.75% | -9.47 |
| 40 dB / -70 dBFS | 36.3% | 99.99% | -10.01 |

The absolute -60 dBFS floor dominates the first three conditions, making their masks identical; they are
not three independent robustness checks. Both additional fixed absolute floors retain the negative gain.
All conditions are reported. Energy retention is not a perceptual rating, and no listener study was run.
