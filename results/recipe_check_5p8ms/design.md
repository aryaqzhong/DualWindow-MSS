# Exploratory learning-rate check for the 5.8 ms wide short-window control

Written on 2026-11-10, before the new runs started. This check is post hoc. It was motivated by the
2.9 ms check's validation logs, which were seen first: at lr 3e-4 the wide E reached best validation
3.553 / 3.618 dB (seeds 0 / 1), against 2.881 / 2.997 dB at the original 1e-3 and 3.368 / 3.392 dB
for the original width-256 E. The test set has already informed this direction, so the result is
exploratory, not confirmatory.

- **Candidate.** `B_sym256_compute` (GRU 776, 5.853 MMAC/frame) at lr 3e-4
  (`configs/B_sym256_compute_lr3e4.yaml`), seeds 0 and 1, 200,000 updates. All other settings are those
  of `B_sym256_compute`. The existing lr 1e-3 runs are reused.
- **Selection, validation only.** The rule is the same as in the 2.9 ms check: average, over the two
  seeds, each seed's best validation median SI-SDR, and pick the higher learning rate. Ties within
  1e-9 dB favor the original 1e-3. The choice is written to `selection.json` before any new test
  evaluation.
- **Test.** Only the selected wide-B setting is scored on the 50 test songs. Its evaluation runs on CPU,
  like that of `C_dual`, the comparator. If 1e-3 is selected, the existing evaluations are reused; those
  ran on GPU inside the training job (cuDNN TF32 differences are at most about 0.03 dB per song, see
  `results/failure_analysis_numerics/`).
- **Comparisons.** Per-song SI-SDR, seeds averaged per song. The primary comparison is `C_dual` − the
  selected wide B; the secondary one is the selected wide B − `B_sym256`. Each reports the median with
  a 10,000-draw song bootstrap CI (seed 0), the number of songs where the first is better, and the
  four seed-pair medians.
- **Asymmetry.** Unlike the 2.9 ms check, `C_dual` gets no extra learning-rate candidate here. Only the
  short-window control gets a second chance, so the comparison is conservative toward the dual
  window. A remaining C advantage holds even after the control's extra tuning. If the wide B closes
  the gap, C at 3e-4 is needed before drawing a conclusion.

Run: `sbatch slurm/train.sbatch configs/B_sym256_compute_lr3e4.yaml SEED` for seeds 0 and 1, then
`sbatch --dependency=afterok:<both jobs> slurm/recipe_check_5p8ms.sbatch` (selection, selected-only
CPU test evaluation, `summary.md`).
