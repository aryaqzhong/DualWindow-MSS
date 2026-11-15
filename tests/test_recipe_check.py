"""Guards against test-based recipe selection and incomplete/stale artifacts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dwmss.config import load_config
from dwmss.data import VALIDATION_TRACKS
from scripts import recipe_check as rc


def scores():
    return {family: {base: [3.0, 3.0], extra: [2.0, 2.0]}
            for family, (base, extra) in rc.FAMILIES.items()}


def complete_log(path: Path):
    rows = []
    for step in range(100, 200001, 100):
        rows.append({"step": step, "loss": 0.2, "grad_norm": 0.1, "lr": 0.001})
        if step % 5000 == 0:
            value = 2.0 if step == 5000 else 1.0
            rows.append({"step": step, "val_si_sdr_median": value, "best": 2.0, "improved": step == 5000})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return rows


def evaluation():
    tracks = [f"track{i:02d}" for i in range(50)]
    frame = pd.DataFrame({"track": tracks, "seconds": 120.0, "si_sdr": 3.0,
                          "usdr": 4.0, "si_sdr_mixture": -5.0, "si_sdri": 8.0})
    identity = {"config": "example", "best_step": 5000}
    metadata = {"config": "example", "step": 5000, "n_tracks": 50, "ckpt": "best.pt", "subset": "test"}
    return frame, metadata, identity, tracks


def test_extra_candidates_change_only_learning_rate():
    for base, extra in rc.FAMILIES.values():
        a, b = [load_config(str(rc.REPO / "configs" / f"{name}.yaml")) for name in (base, extra)]
        rc.validate_recipe_pair(a, b)
        bad = copy.deepcopy(b)
        bad["train"]["weight_decay"] = 0.1
        with pytest.raises(ValueError, match="change only"):
            rc.validate_recipe_pair(a, bad)


def test_selection_uses_mean_of_both_seeds_and_ties_keep_original():
    values = scores()
    wide, wide_extra = rc.FAMILIES["wide_short"]
    dual, dual_extra = rc.FAMILIES["dual"]
    # A lucky single seed cannot select a setting whose two-seed mean is lower.
    values["wide_short"][wide_extra] = [5.5, 0.0]
    values["dual"][dual_extra] = [2.0, 4.0]
    assert rc.choose_settings(values) == {"wide_short": wide, "dual": dual}
    values["wide_short"][wide_extra] = [3.2, 3.2]
    assert rc.choose_settings(values)["wide_short"] == wide_extra
    assert rc.choose_settings(values)["dual"] != dual_extra


def test_selection_near_tie_tolerance_is_frozen():
    values = scores()
    base, extra = rc.FAMILIES["wide_short"]
    values["wide_short"][extra] = [3.0 + 5e-10, 3.0 + 5e-10]
    assert rc.choose_settings(values)["wide_short"] == base
    values["wide_short"][extra] = [3.0 + 2e-9, 3.0 + 2e-9]
    assert rc.choose_settings(values)["wide_short"] == extra


@pytest.mark.parametrize("invalid", ([3.0], [3.0, float("nan")]))
def test_selection_rejects_partial_or_nonfinite_seeds(invalid):
    values = scores()
    values["wide_short"][rc.FAMILIES["wide_short"][0]] = invalid
    with pytest.raises(ValueError, match="two finite"):
        rc.choose_settings(values)


def test_log_requires_all_steps_and_consistent_running_best(tmp_path):
    path = tmp_path / "log.jsonl"
    rows = complete_log(path)
    training, validation = rc.validation_log(path)
    assert len(training) == 2000 and len(validation) == 40
    path.write_text("\n".join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match="validation record"):
        rc.validation_log(path)
    path.write_text("\n".join(json.dumps(row) for row in rows + [rows[-1]]))
    with pytest.raises(ValueError, match="validation record"):
        rc.validation_log(path)
    rows[-1]["best"] = 100
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="running validation best"):
        rc.validation_log(path)


def fake_run(tmp_path):
    name = "example"
    run = tmp_path / "runs" / f"{name}_seed0"
    configs = tmp_path / "configs"
    run.mkdir(parents=True)
    configs.mkdir()
    cfg = {"name": name, "train": {"max_steps": 200000}}
    (configs / f"{name}.yaml").write_text(json.dumps(cfg))
    (run / "DONE").write_text("finished 200000 steps\n")
    (run / "meta.json").write_text(json.dumps({"config": cfg, "seed": 0, "valid_tracks": list(VALIDATION_TRACKS)}))
    complete_log(run / "log.jsonl")
    best = {"config": cfg, "seed": 0, "step": 5000, "best": 2.0, "model": {"weight": torch.ones(1)}}
    torch.save(best, run / "best.pt")
    torch.save({**best, "step": 200000}, run / "last.pt")
    return run, best


def test_run_rejects_wrong_seed_and_partial_completion(tmp_path):
    run, checkpoint = fake_run(tmp_path)
    assert rc.verify_run("example", 0, tmp_path)["best_step"] == 5000
    torch.save({**checkpoint, "seed": 1}, run / "best.pt")
    with pytest.raises(ValueError, match="best.pt"):
        rc.verify_run("example", 0, tmp_path)
    torch.save(checkpoint, run / "best.pt")
    (run / "DONE").write_text("finished 5000 steps")
    with pytest.raises(ValueError, match="completion marker"):
        rc.verify_run("example", 0, tmp_path)


def test_evaluation_requires_aligned_full_song_set_and_best_metadata():
    frame, metadata, identity, tracks = evaluation()
    rc.validate_evaluation(frame, metadata, identity, tracks)
    with pytest.raises(ValueError, match="50 unique"):
        rc.validate_evaluation(frame.iloc[:-1], metadata, identity, tracks)
    changed = frame.copy()
    changed.loc[0, "track"] = "unknown"
    with pytest.raises(ValueError, match="tracks differ"):
        rc.validate_evaluation(changed, metadata, identity, tracks)
    with pytest.raises(ValueError, match="metadata"):
        rc.validate_evaluation(frame, {**metadata, "step": 10000}, identity, tracks)


def test_design_detects_stale_or_malformed_content(tmp_path, monkeypatch):
    design = {"file_hashes": {"config": "abc"}}
    (tmp_path / "design.json").write_text(json.dumps(design))
    monkeypatch.setattr(rc, "make_design", lambda: design)
    assert rc.check_design(tmp_path) == design
    monkeypatch.setattr(rc, "make_design", lambda: {"file_hashes": {"config": "changed"}})
    with pytest.raises(ValueError, match="design differs"):
        rc.check_design(tmp_path)
    (tmp_path / "design.json").write_text("not valid JSON")
    with pytest.raises(json.JSONDecodeError):
        rc.check_design(tmp_path)


def test_design_cannot_be_frozen_after_candidate_training_started(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "E_sym128_compute_lr3e4_seed0"
    run.mkdir(parents=True)
    (run / "log.jsonl").write_text("training already started")
    monkeypatch.setattr(rc, "REPO", tmp_path)
    with pytest.raises(ValueError, match="retroactive design freeze"):
        rc.freeze(tmp_path / "out")


def test_selection_is_idempotent_and_rejects_changed_inputs(tmp_path, monkeypatch):
    (tmp_path / "design.json").write_text("{}")
    monkeypatch.setattr(rc, "check_design", lambda out: {})
    def identity(config, seed):
        return {"config": config, "seed": seed, "best_validation_median": 3.0,
                "hashes": {"best.pt": f"{config}-{seed}"}}
    monkeypatch.setattr(rc, "verify_run", identity)
    first = rc.select(tmp_path)
    assert rc.select(tmp_path) == first
    altered = copy.deepcopy(first)
    altered["choices"]["dual"] = rc.FAMILIES["dual"][1]
    (tmp_path / "selection.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="selection or its inputs changed"):
        rc.select(tmp_path)


def test_missing_selection_cannot_be_recreated_after_test_artifacts(tmp_path, monkeypatch):
    (tmp_path / "design.json").write_text("{}")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "partial.csv").write_text("partial")
    monkeypatch.setattr(rc, "check_design", lambda out: {})
    monkeypatch.setattr(rc, "verify_run", lambda config, seed: {"best_validation_median": 3.0})
    with pytest.raises(ValueError, match="artifacts already exist"):
        rc.select(tmp_path)


def test_partial_selected_evaluation_cannot_enter_summary(tmp_path):
    identity = {"config": "example", "seed": 0}
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "example_seed0_test.csv").write_text("partial")
    assert rc.selected_cache(tmp_path, {}, identity, []) is None
    frame, metadata, identity_base, tracks = evaluation()
    identity = {**identity_base, "seed": 0, "hashes": {"best.pt": "frozen"}}
    path = tmp_path / "eval" / "example_seed0_test.csv"
    frame.to_csv(path, index=False)
    (tmp_path / "selection.json").write_text("{}")
    metadata.update({"checkpoint_sha256": "stale", "selection_sha256": rc.sha256(tmp_path / "selection.json"),
                     "csv_sha256": rc.sha256(path)})
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hashes differ"):
        rc.selected_cache(tmp_path, {}, identity, tracks)


def test_selection_is_written_before_inference_and_unselected_settings_are_skipped(tmp_path, monkeypatch):
    design = {"test_tracks": [f"track{i:02d}" for i in range(50)],
              "original_evaluations": {f"D_dual_2p9ms_seed{seed}": {"frozen": seed} for seed in rc.SEEDS}}
    (tmp_path / "design.json").write_text(json.dumps(design))
    monkeypatch.setattr(rc, "check_design", lambda out: design)
    def identity(config, seed):
        score = 4.0 if config == "E_sym128_compute_lr3e4" else 3.0
        return {"config": config, "seed": seed, "best_validation_median": score}
    monkeypatch.setattr(rc, "verify_run", identity)
    monkeypatch.setattr(rc, "selected_cache", lambda *args: None)
    monkeypatch.setattr(rc, "original_evaluation", lambda ident, tracks: (None, {"frozen": ident["seed"]}))
    evaluated = []
    def evaluate(out, ident, tracks, device):
        frozen = json.loads((out / "selection.json").read_text())
        assert frozen["choices"]["wide_short"] == ident["config"]
        evaluated.append((ident["config"], ident["seed"]))
    monkeypatch.setattr(rc, "evaluate_one", evaluate)
    monkeypatch.setattr(rc, "summarize", lambda out: None)
    rc.evaluate_selected(tmp_path, "cpu")
    assert evaluated == [("E_sym128_compute_lr3e4", 0), ("E_sym128_compute_lr3e4", 1)]
