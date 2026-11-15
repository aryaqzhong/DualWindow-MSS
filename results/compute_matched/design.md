# Prespecified MAC-matched short-window controls

Widths are chosen from the fixed 256:8:1024 grid by model MACs before the new controls are trained. No quality results or CPU timings enter selection. Complete candidate counts are in `candidates.csv` and `design.json`.

| control | reference | GRU width | parameters | model MMAC/frame | target MMAC/frame | MAC error |
|---|---|---:|---:|---:|---:|---:|
| B_sym256_compute | C_dual | 776 | 5,244,377 | 5.852976 | 5.838000 | +0.2565% |
| E_sym128_compute | D_dual_2p9ms | 800 | 5,521,409 | 5.823664 | 5.838000 | -0.2456% |

Only `model.hidden` changes relative to each original symmetric-window baseline. This also widens the GRU input projection and state-to-mask projection; all other layer settings are fixed.

Both controls use 200,000 updates, seeds 0 and 1, the original data split/crops/batches/augmentation/loss, and the checkpoint with best validation median SI-SDR. Test quality never selects a width or checkpoint.

The primary comparisons are C minus the larger B at 5.8 ms and D minus the larger E at 2.9 ms. Also report the wider controls versus their original width-256 baselines, paired over all 50 full-length mono test songs after averaging seeds. Confidence intervals resample songs; they do not include training-seed variance.

This matches model MACs to within 1%, not parameter count or measured CPU time. The larger short-window models have more parameters; their training wall-clock time can differ. The STFT/iSTFT operation count is outside the matching budget. CPU runtime is an outcome measured afterwards for all six models in one ONNX Runtime job on one CPU thread, with trained seed-0 checkpoints. No claim of CPU-matched or device end-to-end latency follows from this design.

Run `python -I scripts/select_compute_matched.py --check` before training and reporting. The selector refuses to replace a frozen design with a different one.
