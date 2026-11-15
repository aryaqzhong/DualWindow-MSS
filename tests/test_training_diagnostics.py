"""Integrity and interpretation checks for retrospective training diagnostics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "training_diagnostics", Path(__file__).resolve().parents[1] / "scripts" / "analyze_training_diagnostics.py")
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def records():
    """Five validation points with a tied maximum: the first maximum wins."""
    output = []
    scores = {20: 1.0, 40: 3.0, 60: 3.0, 80: 2.0, 100: 2.5}
    best = -float("inf")
    for step in range(10, 101, 10):
        output.append({"step": step, "loss": step / 100, "grad_norm": step / 10,
                       "lr": .001, "wave_l1": .01})
        if step in scores:
            score = scores[step]
            improved = score > best
            best = max(best, score)
            output.append({"step": step, "val_si_sdr_median": score, "val_si_sdr_mean": score - .1,
                           "val_si_sdri_median": score + 5, "best": best, "improved": improved})
    return output


def write_log(path, values):
    path.write_text("\n".join(json.dumps(value) for value in values) + "\n")


def test_complete_cadence_and_earliest_tied_maximum(tmp_path):
    path = tmp_path / "log.jsonl"
    write_log(path, records())
    train, valid = diag.read_log(path, 10, 20, 100)
    assert len(train) == 10 and len(valid) == 5
    assert valid.loc[valid.val_si_sdr_median.idxmax(), "step"] == 40


@pytest.mark.parametrize("mutation,message", [
    ("duplicate", "duplicate"), ("missing", "missing"), ("nonfinite", "non-finite"),
    ("best", "running-best"), ("empty", "empty log"), ("unknown", "ambiguous"),
])
def test_corrupt_logs_fail_clearly(tmp_path, mutation, message):
    values = records()
    if mutation == "duplicate":
        values.insert(1, dict(values[0]))
    elif mutation == "missing":
        values.pop(0)
    elif mutation == "nonfinite":
        values[0]["wave_l1"] = np.nan  # Optional metrics must be checked too.
    elif mutation == "best":
        values[2]["best"] = 99
    elif mutation == "unknown":
        values[0]["val_si_sdr_median"] = 1
    path = tmp_path / "log.jsonl"
    write_log(path, values)
    if mutation == "empty":
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match=message):
        diag.read_log(path, 10, 20, 100)


def test_clipping_counts_preclip_norm_strictly_above_threshold():
    frame = pd.DataFrame({"loss": [1, 2, 3], "grad_norm": [4, 5, 6]})
    result = diag.window_summary(frame, 5)
    assert result["clipped_logged_batches"] == 1
    assert result["clipping_fraction_logged"] == pytest.approx(1 / 3)
    assert result["loss_median"] == 2


def make_run(tmp_path):
    run = tmp_path / "runs" / "B_sym256_seed0"
    eval_dir = tmp_path / "eval"
    run.mkdir(parents=True)
    eval_dir.mkdir()
    config = {"name": "B_sym256", "train": {"max_steps": 100, "log_every": 10, "val_every": 20,
               "grad_clip": 5, "lr": .001, "weight_decay": .01}}
    metadata = {"seed": 0, "config": config, "params": 123, "valid_tracks": list(diag.VALIDATION_TRACKS)}
    (run / "meta.json").write_text(json.dumps(metadata))
    (run / "DONE").write_text("finished 100 steps\n")
    write_log(run / "log.jsonl", records())
    checkpoint = {"config": config, "seed": 0, "step": 40, "best": 3.0}
    torch.save(checkpoint, run / "best.pt")
    evaluation = {"config": "B_sym256", "step": 40, "n_tracks": 50, "ckpt": "best.pt", "subset": "test"}
    (eval_dir / "B_sym256_seed0_test.json").write_text(json.dumps(evaluation))
    return run, eval_dir


def test_run_audit_verifies_checkpoints_and_fixed_windows(tmp_path):
    run, eval_dir = make_run(tmp_path)
    summary, train, valid, metadata, sources = diag.analyze_run(run, eval_dir, "B_sym256", 0, 100)
    assert summary["best_step"] == summary["evaluation_step"] == 40
    assert summary["best_to_final_val_drop_db"] == .5
    assert summary["early_sampled_batches"] == summary["late_sampled_batches"] == 2
    assert summary["early_loss_median"] == pytest.approx(.15)
    assert summary["late_loss_median"] == pytest.approx(.95)
    assert summary["clipping_fraction_logged"] == .5
    assert valid.selected_best.sum() == 1
    assert len(sources) == 5 and all(len(source["sha256"]) == 64 for source in sources)


@pytest.mark.parametrize("corruption,message", [
    ("checkpoint", "maximum validation"), ("evaluation", "evaluation does not identify"),
    ("done", "completed 100-update"), ("missing", "missing"), ("split", "14-song"),
])
def test_run_rejects_stale_or_incomplete_artifacts(tmp_path, corruption, message):
    run, eval_dir = make_run(tmp_path)
    if corruption == "checkpoint":
        checkpoint = torch.load(run / "best.pt", weights_only=False)
        checkpoint["step"] = 60  # Equal score, but strict first-maximum selection was step 40.
        torch.save(checkpoint, run / "best.pt")
    elif corruption == "evaluation":
        path = eval_dir / "B_sym256_seed0_test.json"
        evaluation = json.loads(path.read_text())
        evaluation["step"] = 100
        path.write_text(json.dumps(evaluation))
    elif corruption == "done":
        (run / "DONE").write_text("finished 80 steps\n")
    elif corruption == "missing":
        (run / "best.pt").unlink()
    elif corruption == "split":
        path = run / "meta.json"
        metadata = json.loads(path.read_text())
        metadata["valid_tracks"] = ["duplicate"] * 14
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        diag.analyze_run(run, eval_dir, "B_sym256", 0, 100)


def test_bins_do_not_change_clipping_denominator():
    frame = pd.DataFrame({"config": ["B"] * 3, "seed": [0] * 3, "step": [100, 5000, 5100],
                          "loss": [1, 3, 10], "grad_norm": [2, 8, 4], "lr": [.001] * 3,
                          "clipped_logged": [False, True, False]})
    bins = diag.binned_training(frame)
    assert bins.bin_end_step.tolist() == [5000, 10000]
    assert bins.sampled_batches.tolist() == [2, 1]
    assert bins.loss_median.tolist() == [2, 10]
    assert bins.clipping_fraction_logged.tolist() == [.5, 0]
