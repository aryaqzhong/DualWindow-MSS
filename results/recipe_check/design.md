# Bounded learning-rate check at 2.9 ms

Design fixed before new training; summary.md contains completed results when available.

This exploratory follow-up was motivated by the original test results and validation diagnostics. It is not an independent confirmation on a new test set.

Two architectures: short-window GRU-800 (E_sym128_compute) and dual-window GRU-256 (D_dual_2p9ms). Both receive one identical extra learning-rate candidate: 0.0003, alongside the existing 0.001. Weight decay stays 0.01; all other settings, data, 200000 updates, seeds 0/1, validation-checkpoint rule and schedule stay fixed. No architecture changes.

Choose one learning rate per architecture by the arithmetic mean across the two seeds of each seed's own best validation median SI-SDR. Ties within 1e-9 dB favor the original learning rate. The choices and all input hashes are written atomically to selection.json before new test evaluation. Only the selected setting is evaluated on all 50 full test songs. Already selected original settings reuse their frozen evaluation after metadata checks.

Each selected setting averages seeds within songs before song medians and paired comparisons. Bootstrap CIs resample 50 songs 10000 times with seed 0; they do not include training-seed uncertainty. This one-learning-rate check does not establish optimized architectures or general computational optimality.

design.json freezes resolved configurations, all inherited configuration files, source code, original logs/checkpoints/evaluations and dataset path/size/mtime fingerprints. New runs and outputs are separate from the original MAC-matched experiment.
