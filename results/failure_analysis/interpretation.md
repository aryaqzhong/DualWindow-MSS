# Interpretation of the completed failure diagnostics

This is a post-hoc diagnostic of the existing 50-song test set. It retains every song and both seeds, applies five fixed reference-only activity conditions, and does not replace the original full-song evaluation. Numerical definitions, completeness checks and hashes are in [summary.md](summary.md) and [summary_provenance.json](summary_provenance.json).

## The broad median gain survives reference-active scoring

| latency pair | full-song paired median [95% CI] | reference-active paired median [95% CI] | full / active positive songs |
|---|---|---|---|
| C − B at 5.8 ms | +0.76 [+0.57, +0.90] | +0.76 [+0.61, +0.89] | 45/50 / 46/50 |
| D − E at 2.9 ms | +0.73 [+0.59, +0.88] | +0.76 [+0.60, +0.88] | 45/50 / 45/50 |

Across all five fixed activity conditions, C−B median gains range from +0.753 to +0.761 dB and D−E from +0.762 to +0.768 dB; all comparisons cover 50 songs. These are correlated diagnostics on the same recordings, not five independent replications.

The full-song D−E mean remains +0.35 dB with a 95% song-bootstrap interval of [−0.57, +0.91]; the severe lower tail remains part of the primary result. The active-sample metric addresses a different question. In particular, Walkie Talkie changes from negative to positive for both pairs, while D−E on Ben Carrigan / We’ll Talk About It All Tonight changes from +0.001 to −0.270 dB. Equal total win counts do not imply the same winning songs.

## The five original losses at each latency show different trade-offs

All deltas below are dual minus short, with two seed metrics averaged first. Active scoring uses the primary reference-only condition (peak −40 dB, floor −60 dBFS). A positive vocal-energy delta means more vocal energy retained; positive accompaniment attenuation means stronger suppression. The vocal-shape score is SI-SDR of the preserved-vocal component, evaluated against the clean vocal. It is separate from residual accompaniment.

| pair | track | full SI-SDR gain | active gain | active vocal-energy delta | active acc-attenuation delta | active preserved-vocal SI-SDR delta |
|---|---|---:|---:|---:|---:|---:|
| C − B at 5.8 ms | PR - Oh No | -2.73 | -1.37 | -0.23 | -1.82 | +0.90 |
| C − B at 5.8 ms | PR - Happy Daze | -2.25 | -1.47 | +0.32 | -0.41 | -2.11 |
| C − B at 5.8 ms | Detsky Sad - Walkie Talkie | -0.29 | +0.33 | +0.76 | -1.48 | +2.35 |
| C − B at 5.8 ms | The Easton Ellises - Falcon 69 | -0.20 | -0.08 | +1.46 | -2.34 | +1.54 |
| C − B at 5.8 ms | Buitraker - Revo X | -0.10 | -0.09 | -0.15 | +1.32 | -2.03 |
| D − E at 2.9 ms | PR - Happy Daze | -20.06 | -10.15 | -14.78 | +15.09 | -10.97 |
| D − E at 2.9 ms | Detsky Sad - Walkie Talkie | -1.18 | +0.29 | +0.63 | -1.40 | +2.35 |
| D − E at 2.9 ms | PR - Oh No | -1.14 | -1.52 | -0.14 | -2.59 | +0.11 |
| D − E at 2.9 ms | Moosmusic - Big Dummy Shake | -0.41 | -0.17 | +0.30 | -0.26 | +0.23 |
| D − E at 2.9 ms | Buitraker - Revo X | -0.04 | -0.04 | +0.28 | -0.21 | +0.21 |

- **PR - Happy Daze, D/E:** The vocal is 26.06 dB below accompaniment. Primary activity selects 32.6% of samples and retains 99.96% of vocal energy. The full deficit of −20.06 dB decreases to −10.15 dB on active samples. D removes about 15.09 dB more accompaniment but retains 14.78 dB less vocal energy. Its preserved-vocal component also has much worse SI-SDR (D −19.08 vs. E −8.12 dB, seed means), showing shape distortion in addition to attenuation. Inactive output contributes to the extreme full-song score but does not eliminate the remaining active-sample failure.
- **PR - Oh No and Falcon 69:** Several negative gains accompany weaker accompaniment suppression despite equal or better preserved-vocal shape/energy. For example, C preserves 1.46 dB more vocal energy on Falcon 69 but suppresses 2.34 dB less accompaniment. A single blanket explanation of excessive suppression does not fit all failures.
- **Walkie Talkie:** Both full-song losses become small active-sample gains (+0.33 dB for C−B, +0.29 dB for D−E). Better preserved-vocal shape coexists with weaker accompaniment attenuation. Output in reference-inactive intervals matters to the full-song comparison.
- **Revo X:** The mechanisms differ by latency. C suppresses 1.32 dB more accompaniment but has a 2.03 dB worse preserved-vocal shape score. D’s very small full and active losses coexist with slightly greater vocal energy and slightly weaker suppression. These observations motivate further hypotheses; they do not establish a training or architectural cause.

On Happy Daze, uSDR ranks D above E even though both SI-SDR and the preserved-vocal diagnostics favor E. Since an all-zero output is approximately 0 dB uSDR against a nonzero reference, near-zero uSDR together with strong vocal attenuation is not evidence of faithful vocal reproduction. Energy preservation, waveform metrics and projection/correlation must be read together; no perceptual superiority is claimed.

## Representative positive cases were also fixed before these diagnostics

The saved full-song positive gains were ranked, and the 25th/50th/75th ranked quantiles were selected rather than choosing only unusually large improvements. These cases are in `cases.csv` and `cases.json` alongside every original failure.

| pair | positive-gain rank quantile | track | full gain | active gain |
|---|---|---|---:|---:|
| C − B at 5.8 ms | 0.25 | Angels In Amplifiers - I'm Alright | +0.55 | +0.85 |
| C − B at 5.8 ms | 0.5 | BKS - Bulldozer | +0.81 | +1.15 |
| C − B at 5.8 ms | 0.75 | BKS - Too Much | +1.07 | +1.05 |
| D − E at 2.9 ms | 0.25 | Mu - Too Bright | +0.58 | +0.58 |
| D − E at 2.9 ms | 0.5 | M.E.R.C. Music - Knockout | +0.81 | +0.81 |
| D − E at 2.9 ms | 0.75 | Speak Softly - Like Horses | +1.06 | +1.11 |

No listener study or real-microphone measurement was performed. These decompositions diagnose how fixed models allocate vocal preservation and suppression on studio stems; they are not subjective ratings or evidence of acoustic-feedback stability. No MUSDB audio is exported.
