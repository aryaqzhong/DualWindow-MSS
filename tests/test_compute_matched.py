import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dwmss.complexity import describe
from dwmss.config import load_config
from scripts.select_compute_matched import analytic_description, choose_width, make_design
from scripts.summarize_compute_matched import CONFIGS, load_evaluations, paired, sha256, verify_provenance

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["B_sym256", "C_dual", "B_sym256_compute", "E_sym128_compute"])
def test_analytic_counts_match_actual_layers(name):
    cfg = load_config(str(REPO / "configs" / f"{name}.yaml"))
    analytic, actual = analytic_description(cfg), describe(cfg)
    assert analytic["params"] == actual["params"]
    assert analytic["macs_per_frame_M"] == pytest.approx(actual["macs_per_frame_M"], abs=1e-12)


def test_fixed_grid_selects_nearest_width_within_budget():
    for p, width in zip(make_design()["pairs"], [776, 800]):
        cfg = load_config(str(REPO / "configs" / f"{p['baseline']}.yaml"))
        chosen, candidates = choose_width(cfg, p["reference_description"]["macs_per_frame"])
        assert chosen["hidden"] == width
        assert abs(chosen["relative_mac_error"]) < 0.01
        assert abs(chosen["relative_mac_error"]) == min(abs(r["relative_mac_error"]) for r in candidates)


def test_partial_seed_is_not_reported_as_complete(tmp_path):
    name = CONFIGS[0]
    stem = tmp_path / f"{name}_seed0_test"
    pd.DataFrame({"track": [f"song{i}" for i in range(50)], "si_sdr": np.ones(50), "si_sdri": np.ones(50)}).to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".json").write_text(json.dumps({"config": name, "n_tracks": 50, "ckpt": "best.pt"}))
    complete, missing = load_evaluations(tmp_path)
    assert name not in complete
    assert f"{name}_seed1_test" in missing


def test_paired_comparison_aligns_tracks_and_rejects_missing_songs():
    a = pd.Series(np.arange(50.0), index=[f"song{i}" for i in range(50)])
    b = (a - 0.5).iloc[::-1]
    result = paired(a, b)
    assert result["median"] == pytest.approx(0.5)
    assert result["median_ci_lo"] == pytest.approx(0.5)
    assert result["median_ci_hi"] == pytest.approx(0.5)
    assert result["songs_first_better"] == 50
    with pytest.raises(ValueError, match="50 songs"):
        paired(a, b.iloc[:-1])


def test_provenance_detects_changed_checkpoint_or_results(tmp_path):
    checkpoint, csv_path, meta_path = [tmp_path / name for name in ("best.pt", "scores.csv", "scores.json")]
    checkpoint.write_bytes(b"checkpoint at evaluation")
    csv_path.write_text("track,si_sdr\na,1.0\n")
    metadata = {"step": 145000}
    meta_path.write_text(json.dumps(metadata))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"checkpoint_path": str(checkpoint.resolve()), "evaluation_step": 145000,
                                      "checkpoint_sha256": sha256(checkpoint), "csv_sha256": sha256(csv_path),
                                      "json_sha256": sha256(meta_path)}))
    verify_provenance(provenance, checkpoint, csv_path, meta_path, metadata)
    checkpoint.write_bytes(b"changed after evaluation")
    with pytest.raises(ValueError, match="changed after evaluation"):
        verify_provenance(provenance, checkpoint, csv_path, meta_path, metadata)
