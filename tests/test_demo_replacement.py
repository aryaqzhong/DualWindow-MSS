"""Targeted song replacement is complete, source-backed and transactional."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from scripts import build_demo as demo


def source_fixture(tmp_path, monkeypatch, with_acc=True):
    monkeypatch.setattr(demo, "SR", 100)
    monkeypatch.setattr(demo, "EXCERPT_S", 2.0)
    directory = tmp_path / "new_song"
    directory.mkdir()
    rng = np.random.default_rng(2)
    vocal = rng.normal(0, 0.2, 10000).astype(np.float32)
    accompaniment = rng.normal(0, 0.2, 10000).astype(np.float32)
    signals = {"mix.wav": vocal + accompaniment,
               "voc.wav": demo.shift(vocal, -8, 10100),
               "acc.wav": demo.shift(accompaniment, 12, 10100)}
    roles = {"mix.wav": "mix", "voc.wav": "vocals"}
    if with_acc:
        roles["acc.wav"] = "accompaniment"
    files = {}
    for name, role in roles.items():
        path = directory / name
        path.write_bytes(name.encode())
        files[name] = {"role": role, "source": f"https://example.test/{name}", "sha256": demo.file_sha256(path)}
    provenance = {"title": "New song", "artist": "Artist", "license": "Attribution",
                  "license_url": "https://creativecommons.org/licenses/by/3.0/", "page": "https://example.test/song",
                  "files": files}
    (directory / "provenance.json").write_text(json.dumps(provenance))
    monkeypatch.setattr(demo, "load_mono", lambda path: signals[path.name])
    monkeypatch.setattr(demo.sf, "info", lambda path: SimpleNamespace(samplerate=100, channels=1, frames=len(signals[Path(path).name])))
    return directory, provenance, vocal, accompaniment


def test_inspection_aligns_accompaniment_independently_and_records_lengths(tmp_path, monkeypatch):
    directory, _, vocal, accompaniment = source_fixture(tmp_path, monkeypatch)
    _, audio, report = demo.inspect_song_source(directory, required_stems=True)
    assert report["alignment"]["vocals"]["lag_samples"] == 8
    assert report["alignment"]["accompaniment"]["lag_samples"] == -12
    np.testing.assert_allclose(audio["vocals"][20:-20], vocal[20:-20])
    np.testing.assert_allclose(audio["accompaniment"][20:-20], accompaniment[20:-20])
    assert report["sources"]["acc.wav"]["decoded_duration_s"] == 101.0
    assert len(report["alignment"]["accompaniment"]["local_probes"]) == 3


def test_default_excerpt_remains_max_reference_energy_and_explicit_choice_is_sample_rounded(tmp_path, monkeypatch):
    directory, _, _, _ = source_fixture(tmp_path, monkeypatch)
    _, audio, default = demo.inspect_song_source(directory, required_stems=True)
    assert default["excerpt_start_s"] == demo.pick_excerpt(audio["vocals"]) / demo.SR
    assert default["excerpt_selection"] == {"mode": "max_reference_vocal_energy"}
    _, explicit_audio, explicit = demo.inspect_song_source(directory, required_stems=True, excerpt_start_s=10.125)
    assert explicit["excerpt_start_s"] == round(10.125 * demo.SR) / demo.SR
    assert explicit["excerpt_selection"] == {"mode": "audited_explicit", "requested_start_s": 10.125}
    assert explicit["excerpt_s"] == default["excerpt_s"] == demo.EXCERPT_S
    assert len(explicit_audio["mix"]) == len(audio["mix"])  # Inspection never truncates recurrent context.


@pytest.mark.parametrize("start", [-1.0, 4.99, 99.0, float("nan"), float("inf")])
def test_explicit_excerpt_rejects_missing_context_out_of_range_and_nonfinite_values(tmp_path, monkeypatch, start):
    directory, _, _, _ = source_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit excerpt"):
        demo.inspect_song_source(directory, required_stems=True, excerpt_start_s=start)


def test_saved_explicit_excerpt_is_replayed_and_tampering_is_rejected(tmp_path, monkeypatch):
    directory, _, _, _ = source_fixture(tmp_path, monkeypatch)
    _, _, report = demo.inspect_song_source(directory, required_stems=True, excerpt_start_s=10.125)
    saved = json.loads(json.dumps(report))
    _, _, replayed = demo.inspect_reviewed_song_source(directory, saved)
    assert replayed == saved
    changed = copy.deepcopy(saved)
    changed["excerpt_start_s"] += 1
    with pytest.raises(ValueError, match="excerpt differ"):
        demo.inspect_reviewed_song_source(directory, changed)
    changed = copy.deepcopy(saved)
    changed["excerpt_selection"]["requested_start_s"] += 1
    with pytest.raises(ValueError, match="excerpt differ"):
        demo.inspect_reviewed_song_source(directory, changed)


def test_render_keeps_full_continuous_inference_then_crops_explicit_interval(tmp_path, monkeypatch):
    directory, provenance, _, _ = source_fixture(tmp_path, monkeypatch)
    _, audio, report = demo.inspect_song_source(directory, required_stems=True, excerpt_start_s=10.125)
    monkeypatch.setattr(demo, "MODELS", [("example", "Example", 2.9)])
    observed, clips = [], {}
    def separate(model, stft, mixture):
        observed.append(len(mixture))
        return mixture
    monkeypatch.setattr(demo, "separate", separate)
    monkeypatch.setattr(demo, "write_mp3", lambda path, clip: clips.__setitem__(path.name, clip.copy()))
    demo.render_song(directory, tmp_path, provenance, audio, report, {"example": (object(), None)}, None, "cpu")
    assert observed == [len(audio["mix"]), len(audio["mix"])]
    start = int(round(report["excerpt_start_s"] * demo.SR))
    np.testing.assert_array_equal(clips["mix.mp3"], audio["mix"][start:start + int(demo.EXCERPT_S * demo.SR)])


def test_required_stems_never_construct_accompaniment_by_subtraction(tmp_path, monkeypatch):
    directory, _, _, _ = source_fixture(tmp_path, monkeypatch, with_acc=False)
    with pytest.raises(ValueError, match="supplied accompaniment"):
        demo.inspect_song_source(directory, required_stems=True)


def test_inspection_rejects_changed_download_hash(tmp_path, monkeypatch):
    directory, _, _, _ = source_fixture(tmp_path, monkeypatch)
    (directory / "voc.wav").write_bytes(b"changed download")
    with pytest.raises(ValueError, match="hash/URL"):
        demo.inspect_song_source(directory, required_stems=True)


def test_reviewed_alignment_allows_only_roundoff_not_changed_lags_or_sources():
    report = {"lag_samples": 8, "correlation": 0.7, "source_sha256": "a" * 64}
    assert demo.inspection_matches(report, {**report, "correlation": 0.7 + 1e-12})
    assert not demo.inspection_matches(report, {**report, "correlation": 0.701})
    assert not demo.inspection_matches(report, {**report, "lag_samples": 12})
    assert not demo.inspection_matches(report, {**report, "source_sha256": "b" * 64})


def test_targeted_replacement_disables_tf32_and_restores_settings_on_failure():
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError, match="simulated"):
            with demo.replacement_precision():
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                raise RuntimeError("simulated build failure")
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


@pytest.mark.parametrize("missing", ["D_dual_2p9ms", "htdemucs"])
def test_required_render_fails_before_assets_when_any_model_is_missing(tmp_path, missing):
    models = {config: object() for config in demo.BLEED_MODELS}
    reference = object()
    if missing == "htdemucs":
        reference = None
    else:
        models[missing] = None
    with pytest.raises(FileNotFoundError, match="all A–E checkpoints"):
        demo.render_song(Path("new_song"), tmp_path, {}, {}, {}, models, reference, "cpu", True)
    assert not list(tmp_path.iterdir())


def staged_song(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    for slug in ("one", "falling", "three", "four"):
        (audio / slug).mkdir()
        (audio / slug / "original.mp3").write_bytes(slug.encode())
    original = {"sample_rate": demo.SR, "models": [{"id": "unchanged", "si_sdr": 4.02}], "custom": "preserved",
                "songs": [{"slug": slug, "custom": slug} for slug in ("one", "falling", "three", "four")]}
    manifest = tmp_path / "demo.json"
    manifest.write_text(json.dumps(original))
    stage = tmp_path / "staging" / "new_song"
    stage.mkdir(parents=True)
    full_ids = ["mix", *demo.BLEED_MODELS, "htdemucs", "vocal_stem"]
    bleed_ids = ["bleed_input", *(f"bleed_{config}" for config in demo.BLEED_MODELS), "vocal_stem"]
    song = {"slug": "new_song", "title": "New song", "artist": "Artist", "excerpt_s": demo.EXCERPT_S,
            "excerpt_start_s": 10.0, "source_provenance_sha256": "a" * 64,
            "versions": [{"id": version} for version in full_ids], "bleed": [{"id": version} for version in bleed_ids]}
    for version in set(full_ids + bleed_ids):
        for suffix in (".mp3", ".png"):
            (stage / f"{version}{suffix}").write_bytes((version + suffix).encode())
    build = {"source_inspection": {"slug": "new_song", "sample_rate": demo.SR, "source_provenance_sha256": "a" * 64,
                                   "excerpt_s": demo.EXCERPT_S, "excerpt_start_s": 10.0,
                                   "attribution": {"title": "New song", "artist": "Artist"},
                                   "sources": {role: {"role": role} for role in ("mix", "vocals", "accompaniment")}},
             "model_checkpoints": {config: "b" * 64 for config in demo.BLEED_MODELS},
             "numerics": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "dtype": "float32"},
             "asset_sha256": {path.name: demo.file_sha256(path) for path in stage.iterdir()}}
    (stage / "build_provenance.json").write_text(json.dumps(build))
    monkeypatch.setattr(demo, "asset_complete", lambda path, seconds: path.is_file() and path.with_suffix(".png").is_file())
    return manifest, audio, original, stage, song


def test_publish_changes_only_requested_slot_and_preserves_existing_assets(tmp_path, monkeypatch):
    manifest, audio, original, stage, song = staged_song(tmp_path, monkeypatch)
    old_files = {str(path): path.read_bytes() for path in audio.rglob("*") if path.is_file()}
    updated = demo.publish_replacement(manifest, audio, "falling", song, stage, manifest.read_bytes())
    assert [entry["slug"] for entry in updated["songs"]] == ["one", "new_song", "three", "four"]
    assert updated["models"] == original["models"] and updated["custom"] == original["custom"]
    for i in (0, 2, 3):
        assert updated["songs"][i] == original["songs"][i]
    assert all(Path(path).read_bytes() == value for path, value in old_files.items())
    assert json.loads(manifest.read_text()) == updated


def test_publish_rolls_back_new_assets_if_manifest_swap_fails(tmp_path, monkeypatch):
    manifest, audio, _, stage, song = staged_song(tmp_path, monkeypatch)
    original_bytes = manifest.read_bytes()
    replace = os.replace
    def fail_manifest(source, destination):
        if Path(destination) == manifest:
            raise OSError("simulated manifest swap failure")
        return replace(source, destination)
    monkeypatch.setattr(os, "replace", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        demo.publish_replacement(manifest, audio, "falling", song, stage, original_bytes)
    assert manifest.read_bytes() == original_bytes
    assert stage.is_dir() and not (audio / "new_song").exists()
    assert (audio / "falling" / "original.mp3").read_bytes() == b"falling"


@pytest.mark.parametrize("damage", ["missing_png", "missing_model", "wrong_provenance", "changed_asset"])
def test_incomplete_or_wrong_replacement_never_publishes(tmp_path, monkeypatch, damage):
    manifest, audio, _, stage, song = staged_song(tmp_path, monkeypatch)
    before = manifest.read_bytes()
    if damage == "missing_png":
        (stage / "bleed_D_dual_2p9ms.png").unlink()
    elif damage == "missing_model":
        song["versions"] = [entry for entry in song["versions"] if entry["id"] != "htdemucs"]
    elif damage == "wrong_provenance":
        song["title"] = "An unrelated track"
    else:
        (stage / "mix.mp3").write_bytes(b"changed")
    with pytest.raises(ValueError):
        demo.publish_replacement(manifest, audio, "falling", song, stage, before)
    assert manifest.read_bytes() == before and not (audio / "new_song").exists()


def test_concurrent_manifest_changes_are_preserved(tmp_path, monkeypatch):
    manifest, audio, original, stage, song = staged_song(tmp_path, monkeypatch)
    before = manifest.read_bytes()
    changed = copy.deepcopy(original)
    changed["custom"] = "concurrent change"
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed during rendering"):
        demo.publish_replacement(manifest, audio, "falling", song, stage, before)
    assert json.loads(manifest.read_text()) == changed and not (audio / "new_song").exists()


def test_asset_validation_decodes_audio_and_rejects_truncated_png(tmp_path):
    path = tmp_path / "clip.mp3"
    signal = np.sin(np.arange(4410) * 2 * np.pi * 440 / 44100).astype(np.float32) * 0.1
    sf.write(path, signal, 44100, format="MP3", subtype="MPEG_LAYER_III")
    Image.new("RGB", (10, 10), "white").save(path.with_suffix(".png"))
    assert demo.asset_complete(path, 0.1)
    path.with_suffix(".png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert not demo.asset_complete(path, 0.1)
