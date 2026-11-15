# Exploratory learning-rate check for the original short-window baselines

Written on 2026-11-11, before the new runs started. This check is post hoc. The matched-MAC study showed
that the wider short-window models were sensitive to the learning rate (`results/recipe_check/`,
`results/recipe_check_5p8ms/`), but the original width-256 baselines B and E, on which the headline
gains of +0.76 dB (5.8 ms) and +0.73 dB (2.9 ms) rest, were trained at 1e-3 only. The test set has
already informed this question, so the result is exploratory, not confirmatory.

- **Candidates.** `B_sym256` and `E_sym128` at lr 3e-4 (`configs/B_sym256_lr3e4.yaml`,
  `configs/E_sym128_lr3e4.yaml`), seeds 0 and 1, 200,000 updates. All other settings are those of the
  original configs. The existing lr 1e-3 runs are reused.
- **Selection, validation only.** The rule of the earlier checks: for each baseline, average over the
  two seeds each seed's best validation median SI-SDR and pick the learning rate with the higher mean.
  Ties within 1e-9 dB keep the original 1e-3. The choice is written to `selection.json` before any new
  test evaluation.
- **Test.** Training does not score the test songs. Only a newly selected setting is evaluated on the 50
  test songs, on CPU. If 1e-3 is selected for a baseline, its existing evaluation is reused and the
  headline comparison for that latency is unchanged.
- **Comparisons.** Per-song SI-SDR, seeds averaged per song. Primary: `C_dual` − selected B and
  `D_dual_2p9ms` − selected E. Secondary: selected − original baseline. Each reports the median and the
  mean of the per-song differences with 10,000-draw song bootstrap CIs (seed 0), the number of songs
  where the first is better, and the four seed-pair medians.
- **Asymmetry.** `C_dual` gets no second learning rate; `D_dual_2p9ms` already had one and kept 1e-3
  (`results/recipe_check/`). Only the baselines get a second chance here, so the comparison is
  conservative toward the dual window. If a baseline closes the gap, the matching dual-window model at
  3e-4 is needed before drawing a conclusion.
- **Reporting.** Whatever is selected, the README reports the outcome next to the headline numbers.

Run: `sbatch slurm/train.sbatch configs/B_sym256_lr3e4.yaml SEED` and
`sbatch slurm/train.sbatch configs/E_sym128_lr3e4.yaml SEED` for seeds 0 and 1, then
`sbatch --dependency=afterok:<all four jobs> slurm/baseline_lr_check.sbatch` (selection, CPU test
evaluation of newly selected settings only, `summary.md`).
