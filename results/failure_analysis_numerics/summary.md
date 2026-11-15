# CPU/GPU numerical audit for frozen failure diagnostics

The original A–E full-song evaluation used CPU. A strict 0.02 dB agreement guard stopped the first GPU diagnostic attempt on C seed 0 / PR - Happy Daze. This audit uses the same frozen best checkpoints and complete continuous song; it does not modify the original evaluation files.

| run | saved CPU SI-SDR | CPU repeat | default CUDA | CUDA with TF32 disabled | default-minus-CPU | full-precision-minus-CPU |
|---|---:|---:|---:|---:|---:|---:|
| C_dual_seed0 | -25.755736 | -25.755736 | -25.785089 | -25.755179 | -0.029352 | +0.000557 |
| C_dual_seed1 | -30.482350 | -30.482350 | -30.476703 | -30.482236 | +0.005647 | +0.000114 |

Default backend flags were `matmul_tf32=False` and `cudnn_tf32=True`. The full-precision condition sets both to False. CUDA repeats were sample-identical, as were the original CUDA separator and the diagnostic mixture output. Thus the disagreement is associated with CPU/CUDA numerical behavior under the cuDNN TF32 setting, not an error in the diagnostic decomposition.

| run | default CUDA relative waveform L2 error vs CPU | TF32-disabled error | original GPU vs decomposition max absolute difference |
|---|---:|---:|---:|
| C_dual_seed0 | 0.00200452 | 0.00007829 | 0.0 |
| C_dual_seed1 | 0.00137319 | 0.00003782 | 0.0 |

Protocol v2 disables CUDA matmul and cuDNN TF32, records precision flags in runtime and cache provenance, and reruns the full diagnostic from fresh caches. The 0.02 dB agreement tolerance is unchanged. Failed-attempt artifacts and source are retained in `../failure_analysis/attempt_v1_failed/`. These backend differences do not revise the original CPU test results.

Machine-readable rows are in `audit.csv` and `audit.json`. No audio is exported.
