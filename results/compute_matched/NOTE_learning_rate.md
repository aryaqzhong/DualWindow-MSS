# Note: the quality columns of `summary.md` are superseded

`summary.md` in this folder reports the matched-MAC comparison under the shared training recipe (lr 1e-3):
dual − wider short window +1.39 dB at 5.8 ms and +1.56 dB at 2.9 ms. Those wider models trained badly at that
rate. With a learning rate chosen on validation scores (3e-4 for both wider models), the gaps are +0.51 dB
[+0.40, +0.65] and +0.50 dB [+0.28, +0.63] (paired medians; the mean differences are not significant). See
[`../recipe_check_5p8ms/summary.md`](../recipe_check_5p8ms/summary.md) and [`../recipe_check/summary.md`](../recipe_check/summary.md).

`summary.md` is kept as it was written, except that the machine name and job number were removed from its
environment line before publication (see `../path_redactions.json`); later analyses pin its original hash
(`results/diagnostic_followups/protected_inputs.json`). Its CPU-timing columns remain valid: the learning rate does
not change the architecture that is timed.
