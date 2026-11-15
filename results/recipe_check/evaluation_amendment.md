# Evaluation-only FP32 amendment

Frozen at 2026-11-09T22:23:46.012886+00:00, before validation selection or selected-test artifacts existed.

The original `design.json`, all of its frozen files, training candidates, checkpoints and validation-only selection rule are preserved. This addendum binds a separate launcher and batch script by SHA-256; the original training jobs continue to use the original design checker.

Before new GPU test inference, the launcher disables CUDA matmul and cuDNN TF32. A numerical audit on frozen checkpoints found that this reduces a CPU/CUDA SI-SDR discrepancy from 0.029 dB to 0.00056 dB without changing the signal-processing implementation. The audit, extra source files and original design are bound in `evaluation_amendment.json`.

The launcher calls the unchanged validation selector before test inference. New evaluation metadata records precision flags, amendment and launcher hashes; cached new evaluations must carry matching precision provenance. Previously frozen original test evaluations retain their own original backend. This is a numerical-evaluation amendment, not a new training recipe or a change in model selection.
