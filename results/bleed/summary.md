# Vocal bleed & preservation benchmark (MUSDB18-HQ test, 30 s vocal-rich excerpts, mono)

Medians over songs of seed-averaged values. Vocal loss: output vocal energy relative to the clean vocal (dB, 0 = nothing lost). Bleed attenuation: accompaniment energy removed (dB, higher = less bleed).

Songs: 50. Distinct training runs per config: A_sym1024: 2, B_sym256: 2, C_dual: 2, D_dual_2p9ms: 2, E_sym128: 2.

| config | vocals only: loss (dB) / SI-SDR | accompaniment only: attenuation (dB) | per stem (drums / bass / other) | mix −5 dB: SI-SDR / preserved-vocal SI-SDR / vocal loss / bleed atten. |
|---|---|---:|---|---|
| A_sym1024 | -0.04 / 25.0 | 26.5 | 32.2 / 40.6 / 22.1 | 11.23 / 14.7 / -0.59 / 10.6 |
| B_sym256 | -0.08 / 25.8 | 27.2 | 32.6 / 38.2 / 23.2 | 10.69 / 14.7 / -0.75 / 8.8 |
| C_dual | -0.06 / 24.1 | 25.1 | 32.7 / 36.9 / 22.1 | 10.93 / 16.0 / -0.63 / 9.4 |
| D_dual_2p9ms | -0.08 / 25.5 | 27.9 | 32.3 / 39.0 / 22.6 | 10.59 / 14.9 / -0.74 / 9.3 |
| E_sym128 | -0.08 / 25.5 | 33.9 | 37.3 / 42.8 / 30.6 | 9.98 / 14.2 / -0.87 / 8.5 |

## SI-SDR across bleed levels (gamma = 1)

| config | BVR -15 dB | BVR -5 dB | BVR +5 dB |
|---|---:|---:|---:|
| A_sym1024 | 17.91 | 11.23 | 3.89 |
| B_sym256 | 17.86 | 10.69 | 3.38 |
| C_dual | 18.18 | 10.93 | 3.57 |
| D_dual_2p9ms | 17.31 | 10.59 | 3.52 |
| E_sym128 | 16.98 | 9.98 | 2.75 |

## Vocal loss at equal bleed attenuation

BVR -5 dB. Each block uses the midpoint of its curves' common attainable attenuation range, interpolated along the gamma sweep. Targets are recomputed for each comparison; losses in blocks with different targets are not directly comparable.

### A/B/C reference (9.7 dB attenuation)

| config | vocal loss (dB) |
|---|---:|
| A_sym1024 | -0.47 |
| B_sym256 | -0.94 |
| C_dual | -0.68 |

### All available configurations (9.4 dB attenuation)

| config | vocal loss (dB) |
|---|---:|
| A_sym1024 | -0.44 |
| B_sym256 | -0.88 |
| C_dual | -0.63 |
| D_dual_2p9ms | -0.76 |
| E_sym128 | -1.11 |

### D/E at 2.9 ms (8.9 dB attenuation)

| config | vocal loss (dB) |
|---|---:|
| E_sym128 | -0.97 |
| D_dual_2p9ms | -0.67 |

Figures: `bleed/fig_bleed_tradeoff.png`, `bleed/fig_bleed_octaves.png`. Simulated bleed (digital mixing of studio stems); not a real stage microphone and not acoustic feedback.
