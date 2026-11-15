import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dwmss.failure_analysis import activity_mask, block_power, diagnose_signals, paired_bootstrap, projection_metrics


def test_tail_activity_uses_actual_samples_and_reference_only():
    signal = np.r_[np.ones(10), np.zeros(10), np.full(3, .1)]
    powers, counts = block_power(signal, 10)
    np.testing.assert_allclose(powers, [1, 0, .01])
    assert counts.tolist() == [10, 10, 3]
    mask, threshold, _ = activity_mask(signal, 10, 30)
    assert mask.sum() == 13
    assert len(mask) == len(signal)
    assert threshold == -30


def test_activity_floor_retains_no_silent_samples():
    mask, threshold, _ = activity_mask(np.zeros(25), 10)
    assert not mask.any()
    assert threshold == -60
    mask, threshold, _ = activity_mask(np.full(25, 1e-4), 10)
    assert not mask.any()


def test_activity_threshold_sensitivity_is_nested():
    signal = np.repeat([1, .02, .005, .002, 0], 10)
    masks = [activity_mask(signal, 10, relative)[0] for relative in (30, 40, 50)]
    assert np.all(masks[0] <= masks[1]) and np.all(masks[1] <= masks[2])


def test_projection_distinguishes_scale_and_orthogonal_output():
    ref = np.array([1., 0., -1., 0.])
    scaled = projection_metrics(2 * ref, ref)
    assert scaled['alpha'] == pytest.approx(2)
    assert scaled['cosine'] == pytest.approx(1)
    assert scaled['usdr'] == pytest.approx(0)
    ortho = projection_metrics(np.array([0., 1., 0., -1.]), ref)
    assert ortho['cosine'] == 0
    assert ortho['si_sdr'] < -90


def test_diagnostics_score_active_output_and_inactive_energy_separately():
    vocal = np.r_[np.ones(10), np.zeros(10)]
    accompaniment = np.r_[np.zeros(10), np.ones(10)]
    estimate = vocal + accompaniment
    row = diagnose_signals(vocal, accompaniment, estimate, vocal, accompaniment, block_samples=10)
    assert row['active_fraction'] == .5
    assert row['active_si_sdr'] > row['whole_si_sdr'] + 80
    assert row['active_vocal_loss_db'] == 0
    assert row['inactive_output_energy_fraction'] == pytest.approx(.5)
    assert row['inactive_output_dbfs'] == 0


def test_no_active_reference_does_not_produce_active_score():
    zero, one = np.zeros(20), np.ones(20)
    row = diagnose_signals(zero, one, one, zero, one, block_samples=10)
    assert row['active_samples'] == 0
    assert row['active_si_sdr'] is None
    assert row['active_vocal_loss_db'] is None


def test_diagnostics_reject_mismatched_and_nonfinite_signals():
    x = np.ones(20)
    with pytest.raises(ValueError):
        diagnose_signals(x, x[:-1], x, x, x)
    with pytest.raises(ValueError):
        block_power(np.array([np.nan]), 10)


def test_paired_bootstrap_is_deterministic_and_preserves_outlier():
    values = np.r_[np.ones(49), -20.]
    a = paired_bootstrap(values)
    assert a == paired_bootstrap(values)
    assert a['median'] == 1
    assert a['worst'] == -20
    assert a['mean_lo'] < 0
    with pytest.raises(ValueError):
        paired_bootstrap([np.nan])


def test_cache_rejects_changed_payload_and_identity(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/analyze_failures.py'
    spec = importlib.util.spec_from_file_location('failure_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{'run': 'test', 'track': 'song', 'relative_db': relative, 'floor_dbfs': floor, 'whole_si_sdr': 1.,
             'whole_usdr': 1., 'active_fraction': .5, 'threshold_dbfs': -40.,
             'decomposition_relative_rms': 1e-7, 'active_samples': 10, 'active_si_sdr': 2.}
            for relative, floor in module.ACTIVITY_CONDITIONS]
    payload = {'rows': rows, 'blocks': {}}
    identity = {'checkpoint': 'abc'}
    file = tmp_path / 'cache.json'
    module.write_json(file, {'identity': identity, 'payload': payload, 'payload_sha256': module.digest(payload)})
    assert module.cached(file, identity, 'test', 'song') == payload
    assert module.cached(file, {'checkpoint': 'other'}, 'test', 'song') is None
    payload['rows'][0]['whole_si_sdr'] = 5
    module.write_json(file, {'identity': identity, 'payload': payload, 'payload_sha256': 'old'})
    assert module.cached(file, identity, 'test', 'song') is None


def _script():
    path = Path(__file__).resolve().parents[1] / 'scripts/analyze_failures.py'
    spec = importlib.util.spec_from_file_location('failure_guards', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_averaging_does_not_hide_missing_seed_measurement():
    import pandas as pd
    module = _script()
    rows = [{'config': 'B_sym256', 'track': 'song', 'relative_db': 40., 'floor_dbfs': -60.,
             'seed': seed, 'active_vocal_loss_db': value} for seed, value in [(0, -2.), (1, np.nan)]]
    frame = pd.DataFrame(rows)
    assert np.isnan(module.seed_average(frame).active_vocal_loss_db.iloc[0])
    with pytest.raises(ValueError):
        module.seed_average(frame.iloc[:1])
    with pytest.raises(ValueError):
        module.seed_average(pd.concat([frame.iloc[:1], frame.iloc[:1]]))


def test_frame_validator_rejects_wrong_identity_and_missing_condition():
    import pandas as pd
    module = _script()
    run = 'B_sym256_seed0'
    rows = [{'run': run, 'config': 'B_sym256', 'seed': 0, 'track': 'song',
             'relative_db': relative, 'floor_dbfs': floor, 'whole_si_sdr': 1., 'whole_usdr': 1.,
             'active_samples': 1, 'active_fraction': .5, 'threshold_dbfs': -60.,
             'decomposition_relative_rms': 1e-7, 'active_si_sdr': 1., 'active_usdr': 1.}
            for relative, floor in module.ACTIVITY_CONDITIONS]
    frame = pd.DataFrame(rows)
    module.validate_frame(frame, {'tracks': ['song']}, run, partial=True)
    with pytest.raises(ValueError):
        module.validate_frame(frame.iloc[:-1], {'tracks': ['song']}, run, partial=True)
    frame['seed'] = 1
    with pytest.raises(ValueError):
        module.validate_frame(frame, {'tracks': ['song']}, run, partial=True)


def test_zero_energy_ratio_is_undefined():
    from dwmss.failure_analysis import energy_db_ratio
    assert energy_db_ratio(np.zeros(4), np.zeros(4)) is None
    assert energy_db_ratio(np.ones(4), np.zeros(4)) is None
