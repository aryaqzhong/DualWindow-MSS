import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import bleed_benchmark as bleed
from scripts import build_demo as demo


def protocol_frame(songs=("one", "two"), run="C_dual_seed0", config="C_dual"):
    rows = []
    for song in songs:
        for condition, gamma in sorted(bleed.expected_conditions()):
            row = {"song": song, "config": config, "run": run, "condition": condition, "gamma": gamma,
                   "vocal_loss_db": -0.5, "si_sdr": 10.0, "bleed_atten_db": 8.0,
                   "si_sdr_input": 5.0, "vocal_preserved_si_sdr": 20.0}
            row.update({f"atten_{s}_db": 8.0 for s in bleed.STEMS[1:]})
            row.update({f"loss_{a}_{b}": -0.5 for a, b in bleed.OCTAVES})
            row["bvr_db"] = float(condition.removeprefix("mix_bvr")) if condition.startswith("mix_bvr") else np.nan
            if condition == "vocal_only":
                row["bleed_atten_db"] = np.nan  # unused measurements may be absent
            rows.append(row)
    return pd.DataFrame(rows)


def test_cache_accepts_the_full_protocol_not_a_partial_song_list(tmp_path):
    path = tmp_path / "run.csv"
    protocol_frame().to_csv(path, index=False)
    assert len(bleed.expected_conditions()) == 13
    assert bleed.cached_run_complete(path, ["one", "two"], "C_dual_seed0", "C_dual")
    assert not bleed.cached_run_complete(path, ["one"], "C_dual_seed0", "C_dual")
    assert not bleed.cached_run_complete(path, ["one", "two", "three"], "C_dual_seed0", "C_dual")


@pytest.mark.parametrize("corruption", ["truncated", "duplicate", "song", "gamma", "run", "config", "nan", "text"])
def test_cache_rejects_incomplete_or_mismatched_results(tmp_path, corruption):
    df = protocol_frame()
    if corruption == "truncated":
        df = df.iloc[:-1]
    elif corruption == "duplicate":
        df.iloc[-1] = df.iloc[0]
    elif corruption in {"song", "gamma", "run", "config"}:
        df.loc[df.index[0], corruption] = 7.0 if corruption == "gamma" else "wrong"
    else:
        row = df.index[df.condition == "vocal_only"][0]
        if corruption == "text":
            df["si_sdr"] = df["si_sdr"].astype(object)
        df.loc[row, "si_sdr"] = np.nan if corruption == "nan" else "unfinished"
    path = tmp_path / "run.csv"
    df.to_csv(path, index=False)
    assert not bleed.cached_run_complete(path, ["one", "two"], "C_dual_seed0", "C_dual")


def curve(lo, hi):
    return pd.DataFrame({"bleed_atten_db": [lo, hi], "vocal_loss_db": [-lo / 10, -hi / 10]})


def test_new_configs_get_an_attainable_target_without_changing_abc():
    abc = {"A": curve(0, 10), "B": curve(2, 8), "C": curve(4, 12)}
    target_abc, _ = bleed.common_attenuation(abc)
    assert target_abc == pytest.approx(6.0)
    expanded = {**abc, "D": curve(8, 11), "E": curve(7, 9)}
    target, losses = bleed.common_attenuation(expanded)
    assert target == pytest.approx(8.0)
    assert set(losses) == set(expanded)
    assert all(loss == pytest.approx(-0.8) for loss in losses.values())
    assert bleed.common_attenuation({**abc, "D": curve(13, 15)}) is None
    assert bleed.common_attenuation(abc)[0] == target_abc


def test_bleed_only_adds_de_preserves_full_mix_and_skips_song_without_accompaniment(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "SR", 10)
    monkeypatch.setattr(demo, "EXCERPT_S", 2.0)
    source, audio = tmp_path / "source", tmp_path / "audio"
    existing = [{"id": "bleed_input", "kind": "input"}] + [
        {"id": f"bleed_{cfg}", "kind": "causal", "custom": "preserved"}
        for cfg in ("A_sym1024", "B_sym256", "C_dual")
    ] + [{"id": "vocal_stem", "kind": "reference"}]
    song = {"slug": "with_acc", "excerpt_s": 2.0, "excerpt_start_s": 1.0, "vocal_lag_ms": 0.0,
            "versions": [{"id": "mix"}, {"id": "htdemucs"}], "bleed": existing,
            "license": "unchanged", "custom": {"provenance": "preserved"}}
    falling = {**copy.deepcopy(song), "slug": "falling"}
    falling.pop("bleed")
    original = {"sample_rate": 10, "models": [{"id": "htdemucs", "si_sdr": 8.49}],
                "loudness": "unchanged", "songs": [song, falling]}
    manifest_path = tmp_path / "demo.json"
    manifest_path.write_text(json.dumps(original))
    for slug, has_acc in (("with_acc", True), ("falling", False)):
        directory = source / slug
        directory.mkdir(parents=True)
        files = {"mix.wav": {"role": "mix"}, "voc.wav": {"role": "vocals"}}
        if has_acc:
            files["acc.wav"] = {"role": "accompaniment"}
        (directory / "provenance.json").write_text(json.dumps({"files": files}))
    data = {"mix.wav": 0.6, "voc.wav": 0.5, "acc.wav": 0.1}
    monkeypatch.setattr(demo, "load_mono", lambda path: np.full(40, data[path.name], np.float32))
    monkeypatch.setattr(demo, "best_lag", lambda *args: (0, 1.0))
    generated, separated = {}, []
    monkeypatch.setattr(demo, "load_model", lambda cfg, device: (cfg, None))

    def separate(model, stft, x):
        separated.append(model)
        return x

    monkeypatch.setattr(demo, "separate", separate)
    monkeypatch.setattr(demo, "write_mp3", lambda path, x: generated.__setitem__(str(path), x.copy()))
    names = ["E_sym128", "D_dual_2p9ms"]
    new_ids = {f"bleed_{cfg}" for cfg in names}
    monkeypatch.setattr(demo, "asset_complete", lambda path, seconds: path.stem not in new_ids or str(path) in generated)
    updated = demo.build_bleed_only(source, "cpu", True, names, manifest_path, audio)
    assert separated == names
    assert {path.rsplit("/", 1)[-1] for path in generated} == {f"{name}.mp3" for name in new_ids}
    assert updated["models"] == original["models"]
    assert updated["songs"][0]["versions"] == original["songs"][0]["versions"]
    assert updated["songs"][0]["custom"] == original["songs"][0]["custom"]
    assert updated["songs"][0]["license"] == "unchanged"
    assert [v for v in updated["songs"][0]["bleed"] if v["id"] not in new_ids] == existing
    assert updated["songs"][1] == falling
    # A second run can reuse completed D/E assets without running a model again.
    assert demo.build_bleed_only(source, "cpu", True, names, manifest_path, audio) == updated
    assert separated == names
