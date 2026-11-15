# Bounded learning-rate check at 2.9 ms

Status: complete.

Exploratory follow-up motivated by already observed original test results; this reuses the same test set. The two learning-rate candidates and selection rule were frozen before new training. This is a symmetric one-candidate sensitivity check, not an optimization study.

Each architecture received the existing 0.001 setting plus one 0.0003 setting, each with seeds 0/1 and 200000 updates. Weight decay remains 0.01 and all other settings are identical. Selection uses the mean of the two seeds' own best validation median SI-SDR. One setting is chosen per architecture; ties within 1e-9 dB favor the original. selection.json was written before new test inference.

| architecture | selected setting | learning rate | selection validation score | test SI-SDR | test seed 0 / 1 | best steps 0 / 1 |
|---|---|---:|---:|---:|---|---|
| dual | D_dual_2p9ms | 0.001 | 4.232 | 4.02 | 3.99 / 4.01 | 145000 / 195000 |
| wide_short | E_sym128_compute_lr3e4 | 0.0003 | 3.586 | 3.41 | 3.43 / 3.40 | 175000 / 145000 |

Validation selection score averages seed maxima; test SI-SDR averages seeds within each song before the 50-song median. These aggregation orders are different.

| comparison | median paired gain [95% CI] | mean paired gain [95% CI] | first better |
|---|---|---|---:|
| selected_dual_minus_selected_wide_short | +0.50 [+0.28, +0.63] | +0.06 [-0.82, +0.59] | 41 / 50 |
| original_dual_minus_original_wide_short | +1.56 [+1.19, +1.65] | +1.03 [+0.01, +1.63] | 48 / 50 |
| selected_minus_original_wide_short | +0.95 [+0.74, +1.08] | +0.97 [+0.69, +1.25] | 45 / 50 |
| selected_minus_original_dual | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0 / 50 |

Song bootstrap: 10000 draws, seed 0, 95% percentile intervals. Intervals do not include training-seed uncertainty. All 50 songs remain in the comparison.

The selected checkpoint per seed is validation best.pt, so early stopping was already applied. Changing the learning rate can reveal recipe sensitivity; neither outcome proves the cause of the original validation deterioration, nor excludes better short-window architectures or training strategies. Only 2.9 ms was checked. Architecture, model MACs and algorithmic latency are unchanged, but no new CPU timing or physical end-to-end latency claim is made.

All new evaluations live in this directory; original MAC-matched reports and evaluations remain intact. See design.json, selection.json and evaluation sidecars for source/checkpoint/CSV hashes.
