import json

import numpy as np
import pandas as pd
import pytest
import torch

from dwmss.bleed_paths import causal_filter, energy, filtered_window, fixed_paths, scale_to_bvr
from dwmss.stft import DualWindowSTFT
from scripts.bleed_robustness import (GAMMAS, atomic_json, atomic_text, bootstrap_difference,
                                      completed_frames, equal_attenuation, file_sha256,
                                      digest, read_cache, summarize)


@pytest.mark.parametrize("name", ["dry", "filtered_delay", "synthetic_room"])
def test_paths_do_not_anticipate_future_or_advance_impulses(name):
    h = fixed_paths()[name].impulse
    rng = np.random.default_rng(4)
    a = rng.standard_normal(16000).astype(np.float32)
    b = a.copy()
    b[6000:] = rng.standard_normal(len(b) - 6000)
    np.testing.assert_allclose(causal_filter(a, h)[:6000], causal_filter(b, h)[:6000], atol=2e-7)
    impulse = np.zeros(16000, dtype=np.float32)
    impulse[2000] = 1
    out = causal_filter(impulse, h)
    assert np.max(np.abs(out[:2000])) < 1e-7
    assert np.argmax(np.abs(out)) >= 2000


@pytest.mark.parametrize("name", ["dry", "filtered_delay", "synthetic_room"])
def test_one_shared_path_is_linear_across_accompaniment_stems(name):
    h = fixed_paths()[name].impulse
    rng = np.random.default_rng(5)
    stems = [rng.standard_normal(20000).astype(np.float32) * 0.05 for _ in range(3)]
    mixed = causal_filter(sum(stems), h)
    decomposed = sum(causal_filter(s, h) for s in stems)
    np.testing.assert_allclose(mixed, decomposed, atol=1e-7, rtol=1e-5)


def test_real_path_history_is_kept_before_model_context():
    h = fixed_paths()["synthetic_room"].impulse
    x = np.zeros(22000, dtype=np.float32)
    start, length = 10000, 6000
    x[start - 200] = 1  # arrives during context because the path has a real delay
    history_start = start - (len(h) - 1)
    actual = filtered_window(x[history_start:start + length], h, start - history_start, length)
    expected = causal_filter(x, h)[start:start + length]
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert energy(actual) > 0.9
    assert energy(causal_filter(x[start:start + length], h)) == 0


def test_bvr_normalization_uses_scored_window_and_preserves_common_gain():
    rng = np.random.default_rng(6)
    vocal = rng.standard_normal(18000).astype(np.float32) * 0.1
    vocal[:3000] *= 10  # warmup energy must not determine the gain
    raw = {s: rng.standard_normal(len(vocal)).astype(np.float32) * 0.03 for s in ("drums", "bass", "other")}
    h = fixed_paths()["filtered_delay"].impulse
    stems = {s: causal_filter(x, h) for s, x in raw.items()}
    score = slice(3000, None)
    scaled, gain, achieved = scale_to_bvr(vocal, stems, score)
    assert achieved == pytest.approx(-5.0, abs=1e-5)
    expected_gain = np.sqrt(10 ** (-5 / 10) * energy(vocal[score]) / energy(sum(stems.values())[score]))
    assert gain == pytest.approx(expected_gain)
    for s in stems:
        np.testing.assert_allclose(scaled[s], gain * stems[s], atol=1e-8)
    whole_bvr = 10 * np.log10(energy(sum(scaled.values())) / energy(vocal))
    assert abs(whole_bvr + 5) > 5


def test_silent_bvr_cannot_be_silently_normalized():
    with pytest.raises(ValueError, match="silent"):
        scale_to_bvr(np.zeros(100), {"drums": np.ones(100)}, slice(None))


def test_frozen_paths_are_reproducible_and_provenance_is_exact():
    a, b = fixed_paths(), fixed_paths()
    for name in a:
        np.testing.assert_array_equal(a[name].impulse, b[name].impulse)
        assert a[name].provenance() == b[name].provenance()
        np.testing.assert_array_equal(a[name].provenance()["coefficients"], a[name].impulse)
    assert len(a["dry"].impulse) == 1
    assert np.argmax(abs(a["synthetic_room"].impulse)) == round(0.012 * 44100)


@pytest.mark.parametrize("n_analysis,n_synthesis", [(256, 256), (2048, 256), (2048, 128), (128, 128)])
def test_same_mixture_mask_decomposes_vocal_and_transformed_bleed(n_analysis, n_synthesis):
    rng = np.random.default_rng(7)
    vocal = rng.standard_normal(10000).astype(np.float32) * 0.04
    h = fixed_paths()["synthetic_room"].impulse
    stems = {s: causal_filter(rng.standard_normal(len(vocal)).astype(np.float32) * 0.03, h)
             for s in ("drums", "bass", "other")}
    stems, _, _ = scale_to_bvr(vocal, stems, slice(1000, None))
    stft = DualWindowSTFT(n_analysis, n_synthesis)
    components = [torch.from_numpy(x)[None] for x in [vocal, *stems.values()]]
    mix = sum(components)
    X = stft.analysis(mix)
    mask = torch.sigmoid(torch.log(X.abs() + 0.01))  # a nonconstant mask inferred from this mixture
    direct = stft.synthesis(mask * X, len(vocal))
    decomposed = sum(stft.synthesis(mask * stft.analysis(x), len(vocal)) for x in components)
    torch.testing.assert_close(direct, decomposed, atol=2e-6, rtol=2e-4)


def example_rows(song="song", run="B_sym256_seed0"):
    return [{"song": song, "run": run, "config": "B_sym256", "seed": 0,
             "scenario": p, "gamma": g, "bvr_db": -5.0, "actual_bvr_db": -5.0,
             "si_sdr": 2.0, "si_sdr_input": 1.0, "vocal_loss_db": -g,
             "bleed_atten_db": 4 * g}
            for p in fixed_paths() for g in GAMMAS]


def test_cache_rejects_stale_identity_partial_and_corrupt_results(tmp_path):
    cache = tmp_path / "song.json"
    identity = {"experiment_id": "frozen", "song": "song"}
    rows = example_rows()
    atomic_json(cache, {"complete": True, "identity": identity, "rows": rows, "rows_sha256": digest(rows)})
    assert read_cache(cache, identity, "song", "B_sym256_seed0", fixed_paths()) == rows
    assert read_cache(cache, {**identity, "experiment_id": "changed"}, "song", "B_sym256_seed0", fixed_paths()) is None
    assert read_cache(cache, identity, "song", "B_sym256_seed0", fixed_paths(), "C_dual", 0) is None
    atomic_json(cache, {"complete": True, "identity": identity, "rows": rows[:-1], "rows_sha256": digest(rows[:-1])})
    assert read_cache(cache, identity, "song", "B_sym256_seed0", fixed_paths()) is None
    atomic_text(cache, "{truncated")
    assert read_cache(cache, identity, "song", "B_sym256_seed0", fixed_paths()) is None


def test_equal_attenuation_has_same_target_across_paths_and_never_extrapolates():
    rows = []
    for song in ("shared", "disjoint"):
        for config in ("B_sym256", "C_dual"):
            for i, scenario in enumerate(fixed_paths()):
                for j, gamma in enumerate(GAMMAS):
                    attenuation = 2 * j + (100 if song == "disjoint" and i == 2 else i)
                    rows.append({"song": song, "config": config, "scenario": scenario, "gamma": gamma,
                                 "bleed_atten_db": attenuation,
                                 "vocal_loss_db": -attenuation / 2 + (0.2 if config == "C_dual" else 0)})
    equal = equal_attenuation(pd.DataFrame(rows), list(fixed_paths()))
    assert set(equal.song) == {"shared"}
    assert len(equal) == 6
    assert set(equal.target_bleed_atten_db) == {5.0}
    paired = equal.pivot(index="scenario", columns="config", values="vocal_loss_db")
    np.testing.assert_allclose(paired.C_dual - paired.B_sym256, 0.2)


def test_summary_averages_seeds_before_song_comparisons_and_checks_completion(tmp_path):
    spec = {"protocol_id": "protocol", "paths": [{"name": p} for p in fixed_paths()]}
    excerpts = {"manifest_id": "excerpts", "songs": [{"song": "a"}, {"song": "b"}]}
    atomic_json(tmp_path / "protocol.json", spec)
    atomic_json(tmp_path / "excerpts.json", excerpts)
    names = []
    for config in ("B_sym256", "C_dual"):
        for seed in (0, 1):
            run = f"{config}_seed{seed}"
            names.append(run)
            rows = example_rows("a", run) + example_rows("b", run)
            for row in rows:
                row["config"], row["seed"] = config, seed
                row["si_sdr"] = (1 if seed == 0 else 9) + (2 if config == "C_dual" else 0)
            path = tmp_path / f"{run}.csv"
            atomic_text(path, pd.DataFrame(rows).to_csv(index=False))
            atomic_json(tmp_path / "run_provenance" / f"{run}.json",
                        {"experiment_id": run, "protocol_id": "protocol", "excerpt_manifest_id": "excerpts",
                         "config": config, "seed": seed})
            atomic_json(tmp_path / "completions" / f"{run}.json",
                        {"complete": True, "experiment_id": run, "csv_sha256": file_sha256(path),
                         "rows": len(rows), "songs": 2})
    summarize(tmp_path, names)
    per_song = pd.read_csv(tmp_path / "per_song.csv")
    assert set(per_song[per_song.config == "B_sym256"].si_sdr) == {5.0}
    comparisons = pd.read_csv(tmp_path / "paired_comparisons.csv")
    main = comparisons[(comparisons.kind == "dual_minus_short") & (comparisons.metric == "si_sdr")]
    np.testing.assert_allclose(main["median"], 2.0)
    assert set(main.songs) == {2}
    assert bootstrap_difference([2, 2])["median_ci_low"] == 2
    bad = tmp_path / f"{names[0]}.csv"
    atomic_text(bad, bad.read_text() + "corrupt\n")
    _, missing = completed_frames(tmp_path, spec, excerpts, names)
    assert names[0] in missing
