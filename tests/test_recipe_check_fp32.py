import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/recipe_check_fp32.py'
    spec = importlib.util.spec_from_file_location('recipe_fp32_test', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def setup(tmp_path, module):
    out = tmp_path / 'results/recipe_check'
    out.mkdir(parents=True)
    (out / 'design.json').write_text('{"original": true}\n')
    for name in module.SOURCES:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('frozen additional source')
    for name in ['audit.json', 'summary.md', 'provenance.json']:
        file = tmp_path / 'results/failure_analysis_numerics' / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('audited numerical evidence')
    recipe = SimpleNamespace(check_design=lambda path: {'original': True})
    return tmp_path, out, recipe


def test_freeze_preserves_original_design_and_records_pretest_absence(module, setup):
    repo, out, recipe = setup
    before = (out / 'design.json').read_bytes()
    amendment = module.freeze_amendment(out, recipe, repo)
    assert amendment['pretest_absence_proof']['selection_absent'] is True
    assert amendment['pretest_absence_proof']['selected_test_artifacts_absent'] is True
    assert (out / 'design.json').read_bytes() == before
    assert not (out / 'selection.json').exists()
    assert not (out / 'eval').exists()
    assert module.check_amendment(out, recipe, repo) == amendment


@pytest.mark.parametrize('artifact', ['selection.json', 'eval/test.csv'])
def test_refuses_retroactive_precision_amendment(module, setup, artifact):
    repo, out, recipe = setup
    path = out / artifact
    path.parent.mkdir(exist_ok=True)
    path.write_text('{}')
    with pytest.raises(ValueError, match='retrospective'):
        module.freeze_amendment(out, recipe, repo)


def test_amendment_rejects_changed_additional_source_and_original_design(module, setup):
    repo, out, recipe = setup
    module.freeze_amendment(out, recipe, repo)
    source = repo / module.SOURCES[0]
    old = source.read_bytes()
    source.write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        module.check_amendment(out, recipe, repo)
    source.write_bytes(old)
    (out / 'design.json').write_text('changed design')
    with pytest.raises(ValueError, match='Original frozen design changed'):
        module.check_amendment(out, recipe, repo)


def test_precision_is_disabled_before_inference_and_cache_requires_proof(module, setup):
    repo, out, recipe = setup
    amendment = module.freeze_amendment(out, recipe, repo)
    module.write_json(out / 'selection.json', {'created_utc': amendment['created_utc']})
    fake_torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                            cudnn=SimpleNamespace(allow_tf32=True)), __version__='test')
    identity = {'config': 'selected', 'seed': 0}
    def original_evaluate(path, selected, tracks, device):
        assert fake_torch.backends.cuda.matmul.allow_tf32 is False
        assert fake_torch.backends.cudnn.allow_tf32 is False
        module.write_json(path / 'eval/selected_seed0_test.json', {'config': selected['config'], 'seed': 0})
    recipe.evaluate_one = original_evaluate
    recipe.selected_cache = lambda *args: {'verified_original_cache': True}
    old_evaluator = recipe.evaluate_one
    module.configure_fp32(fake_torch)
    with module.guarded_evaluation(recipe, out, amendment, fake_torch, repo):
        recipe.evaluate_one(out, identity, [], 'cuda')
        assert recipe.selected_cache(out, {}, identity, [])
        path = out / 'eval/selected_seed0_test.json'
        metadata = json.loads(path.read_text())
        metadata['evaluation_precision']['cudnn_allow_tf32'] = True
        module.write_json(path, metadata)
        with pytest.raises(ValueError, match='FP32 amendment provenance'):
            recipe.selected_cache(out, {}, identity, [])
    assert recipe.evaluate_one is old_evaluator


def test_guard_rejects_reenabled_tf32_before_inference(module, setup):
    repo, out, recipe = setup
    amendment = module.freeze_amendment(out, recipe, repo)
    module.write_json(out / 'selection.json', {'created_utc': amendment['created_utc']})
    fake = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                           cudnn=SimpleNamespace(allow_tf32=True)))
    recipe.evaluate_one = lambda *args: pytest.fail('Inference must not start')
    recipe.selected_cache = lambda *args: None
    with module.guarded_evaluation(recipe, out, amendment, fake, repo):
        with pytest.raises(RuntimeError, match='before selected-test inference'):
            recipe.evaluate_one(out, {'config': 'selected', 'seed': 0}, [], 'cuda')


def test_selection_must_not_predate_amendment(module, setup):
    repo, out, recipe = setup
    amendment = module.freeze_amendment(out, recipe, repo)
    module.write_json(out / 'selection.json', {'created_utc': '2000-01-01T00:00:00+00:00'})
    with pytest.raises(ValueError, match='predates'):
        module.check_selection_time(out, amendment)


def test_amendment_rejects_missing_audit_or_foreign_absence_proof(module, setup):
    repo, out, recipe = setup
    amendment = module.freeze_amendment(out, recipe, repo)
    original = json.loads(json.dumps(amendment))
    amendment['numerical_audit_hashes'] = {}
    module.write_json(out / 'evaluation_amendment.json', amendment)
    with pytest.raises(ValueError, match='complete numerical audit'):
        module.check_amendment(out, recipe, repo)
    original['pretest_absence_proof']['checked_selection_path'] = '/different/selection.json'
    module.write_json(out / 'evaluation_amendment.json', original)
    with pytest.raises(ValueError, match='this output directory'):
        module.check_amendment(out, recipe, repo)
