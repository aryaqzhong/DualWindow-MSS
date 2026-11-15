"""Build the listening demo (docs/): separated vocals of CC BY songs for every model, loudness-matched MP3s.

For each song from scripts/fetch_demo_audio.py:
1. Decode the full mix and the stems, resample to 44.1 kHz, average to mono (as for the models).
2. Align the vocal stems and supplied accompaniment independently to the mix by cross-correlation (±5 s).
3. Pick the 24 s excerpt with the most vocal energy; models process the whole song (so the recurrent state is
   warmed up) with the same offline path as the test evaluation, which equals frame-by-frame streaming
   (tests/test_streaming.py).
4. Versions: the mix, each trained causal model (seed 0), the offline HTDemucs reference, and the original
   vocal stem. Songs with an accompaniment stem get a second scenario: simulated vocal-mic bleed (accompaniment
   5 dB below the vocal), separated by A, B, C, D and E.
5. Every clip is loudness-matched (RMS −20 dBFS, peaks limited to −1 dBFS) and written as mono MP3, with a
   spectrogram image (log frequency, so that low-frequency differences between windows are visible).
6. A model table (algorithmic latency, test-set SI-SDR, single-thread real-time factor) from the saved results.

Usage: python -I scripts/build_demo.py [--src data/demo_ccmixter] [--extra-path DIR_WITH_DEMUCS]
       python -I scripts/build_demo.py --table-only   (refresh only the model table in docs/demo.json)
       python -I scripts/build_demo.py --bleed-only --skip-existing (add missing bleed versions; preserve full mix)
       python -I scripts/build_demo.py --inspect-song NEW --required-stems --alignment-out REPORT.json [--excerpt-start SECONDS]
       python -I scripts/build_demo.py --replace-song OLD --song NEW --required-stems --alignment-report REPORT.json --extra-path DIR
Outputs: docs/audio/<song>/<version>.mp3 and .png, docs/demo.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import fftconvolve, resample_poly  # noqa: E402

from dwmss.config import load_config  # noqa: E402
from dwmss.model import MaskNet  # noqa: E402
from dwmss.separator import separate  # noqa: E402
from dwmss.stft import DualWindowSTFT  # noqa: E402

SR = 44100
EXCERPT_S = 24.0
TARGET_RMS_DB, PEAK_DB = -20.0, -1.0
BVR_DB = -5.0
# ccMixter credits the artists sampled in an upload in its "featuring" metadata. Provenance files fetched
# before 2026-11-10 did not record it; this fallback was checked against the ccMixter API on that date.
FEATURING = {"persephone": "Vidian, Dimitri Artemenko"}


def featuring_of(slug: str, prov: dict) -> str:
    return prov.get("featuring") or FEATURING.get(slug, "")


MODELS = [("A_sym1024", "A · long window", 23.2), ("B_sym256", "B · short window", 5.8),
          ("C_dual", "C · dual window", 5.8), ("E_sym128", "E · short window", 2.9),
          ("D_dual_2p9ms", "D · dual window", 2.9)]
BLEED_MODELS = [c for c, _, _ in MODELS]


def load_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        fr = Fraction(SR, sr).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return x


def best_lag(ref: np.ndarray, x: np.ndarray, max_lag_s: float = 5.0, decim: int = 4) -> tuple[int, float]:
    """Lag (samples) that best aligns x to ref (x shifted right by lag), from 90 s of decimated signal."""
    n = min(len(ref), len(x), 90 * SR)
    a, b = ref[:n:decim].astype(np.float64), x[:n:decim].astype(np.float64)
    c = fftconvolve(a, b[::-1], mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    keep = np.abs(lags) <= max_lag_s * SR / decim
    i = int(np.argmax(c[keep]))
    peak = float(c[keep][i] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return int(lags[keep][i] * decim), peak


def shift(x: np.ndarray, lag: int, n: int) -> np.ndarray:
    y = np.zeros(n, np.float32)
    if lag >= 0:
        m = min(n - lag, len(x))
        if m > 0:
            y[lag:lag + m] = x[:m]
    else:
        m = min(n, len(x) + lag)
        if m > 0:
            y[:m] = x[-lag:-lag + m]
    return y


def pick_excerpt(voc: np.ndarray) -> int:
    blk, k = SR, int(EXCERPT_S)
    e = np.array([np.sum(voc[i * blk:(i + 1) * blk] ** 2) for i in range(len(voc) // blk)])
    cs = np.convolve(e, np.ones(k), mode="valid")
    lo = 5  # keep at least 5 s of song before the excerpt
    return (lo + int(np.argmax(cs[lo:]))) * blk


def loudness_match(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9
    y = x * (10 ** (TARGET_RMS_DB / 20) / rms)
    peak = np.max(np.abs(y)) + 1e-9
    lim = 10 ** (PEAK_DB / 20)
    if peak > lim:
        y = y * (lim / peak)
    return y.astype(np.float32)


def write_spectrogram(path: Path, x: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.signal import stft as sp_stft
    f, t, Z = sp_stft(loudness_match(x), fs=SR, nperseg=2048, noverlap=2048 - 256)
    S = 20 * np.log10(np.abs(Z) + 1e-6)
    S = np.clip(S, S.max() - 70, S.max())
    cmap = LinearSegmentedColormap.from_list("ink", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0b0b0b"])
    fig = plt.figure(figsize=(9, 2.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    keep = (f >= 50) & (f <= 12000)
    ax.pcolormesh(t, f[keep], S[keep], cmap=cmap, shading="auto", rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(50, 12000)
    ax.axis("off")
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


def write_mp3(path: Path, x: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), loudness_match(x), SR, format="MP3", subtype="MPEG_LAYER_III", compression_level=0.25)
    write_spectrogram(path.with_suffix(".png"), x)


def model_table() -> list[dict]:
    """Algorithmic latency, test-set SI-SDR (median over songs, seeds averaged) and ONNX real-time factor."""
    import pandas as pd
    rows = []
    bench = REPO / "results" / "latency_benchmark.csv"
    b = pd.read_csv(bench) if bench.exists() else None
    for cfg, label, lat in MODELS + [("htdemucs", "HTDemucs · offline reference", None)]:
        fs = sorted((REPO / "results" / "eval").glob(f"{cfg}_seed*_test.csv")) or \
            sorted((REPO / "results" / "eval").glob(f"{cfg}_test.csv"))
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs]).groupby("track")["si_sdr"].mean()
        r = {"id": cfg, "label": label, "latency_ms": lat, "si_sdr": round(float(d.median()), 2), "rtf": None,
             "p99_ms": None, "hop_ms": None, "deadline_miss": None}
        if b is not None:
            m = b[(b["config"] == cfg) & (b["backend"] == "onnx")]
            if len(m) and str(m["weights"].iloc[0]) != "random":
                m = m.iloc[0]
                r.update({"rtf": round(float(m["rtf"]), 3), "p99_ms": round(float(m["p99_us"]) / 1000, 2),
                          "hop_ms": round(float(m["hop_ms"]), 2), "deadline_miss": float(m["deadline_miss_rate"])})
        rows.append(r)
    return rows


def load_model(cfg_name: str, device):
    run = REPO / "runs" / f"{cfg_name}_seed0" / "best.pt"
    if not run.exists():
        return None
    cfg = load_config(str(REPO / "configs" / f"{cfg_name}.yaml"))
    stft = DualWindowSTFT(cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"]).to(device)
    model = MaskNet(**cfg["model"]).to(device)
    model.load_state_dict(torch.load(run, map_location=device, weights_only=False)["model"])
    model.eval()
    return model, stft


def copy_figures():
    """Figures shown on the demo page (GitHub Pages serves only docs/)."""
    import shutil
    (REPO / "docs" / "img").mkdir(parents=True, exist_ok=True)
    for name in ("fig_paired_gain.png", "fig_quality_latency.png"):
        if (REPO / "results" / name).exists():
            shutil.copy(REPO / "results" / name, REPO / "docs" / "img" / name)


def asset_complete(path: Path, seconds: float) -> bool:
    """Both a decodable clip of the expected length and its PNG must be present."""
    try:
        from PIL import Image
        info = sf.info(str(path))
        with Image.open(path.with_suffix(".png")) as image:
            if image.format != "PNG":
                return False
            image.verify()
        if info.format != "MP3" or info.samplerate != SR or info.channels != 1 or abs(info.duration - seconds) >= 0.1:
            return False
        samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return rate == SR and abs(len(samples) / SR - seconds) < 0.1 and np.isfinite(samples).all()
    except (OSError, RuntimeError, ValueError, SyntaxError, EOFError):
        return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(f"unsafe song slug: {value!r}")
    return value


def catalog_directories(src: Path) -> list[Path]:
    """Use the explicit download catalog, excluding leftover or exploratory folders."""
    from scripts.fetch_demo_audio import SONGS
    slugs = [safe_slug(slug) for slug, _ in SONGS.values()]
    if len(set(slugs)) != len(slugs):
        raise ValueError("the demo download catalog contains duplicate slugs")
    directories = [src / slug for slug in sorted(slugs)]
    for directory in directories:
        if not (directory / "provenance.json").is_file():
            raise FileNotFoundError(f"catalog song has no source provenance: {directory}")
    return directories


def local_alignment_probes(mix: np.ndarray, component: np.ndarray) -> list[dict]:
    """Residual offsets after global alignment; quiet windows remain explicitly unscored."""
    length = min(len(mix), 30 * SR)
    starts = sorted({min(5 * SR, max(0, len(mix) - length)), max(0, (len(mix) - length) // 2),
                     max(0, len(mix) - length)})
    probes = []
    for start in starts:
        signal = component[start:start + length]
        rms_db = float(10 * np.log10(np.mean(signal.astype(np.float64) ** 2) + 1e-30))
        probe = {"start_s": start / SR, "duration_s": len(signal) / SR, "component_rms_dbfs": rms_db,
                 "residual_lag_samples": None, "correlation": None}
        if rms_db >= -60:
            lag, correlation = best_lag(mix[start:start + length], signal, max_lag_s=0.1)
            probe.update({"residual_lag_samples": lag, "correlation": correlation})
        probes.append(probe)
    return probes


def inspection_matches(reviewed, current) -> bool:
    """Exact structure/identities/lags, allowing only floating-point reduction roundoff."""
    if isinstance(reviewed, dict) and isinstance(current, dict):
        return set(reviewed) == set(current) and all(inspection_matches(reviewed[key], current[key]) for key in reviewed)
    if isinstance(reviewed, list) and isinstance(current, list):
        return len(reviewed) == len(current) and all(inspection_matches(a, b) for a, b in zip(reviewed, current))
    if isinstance(reviewed, float) or isinstance(current, float):
        return (isinstance(reviewed, (int, float)) and isinstance(current, (int, float))
                and math.isfinite(reviewed) and math.isfinite(current)
                and math.isclose(reviewed, current, rel_tol=1e-12, abs_tol=1e-10))
    return reviewed == current


@contextmanager
def replacement_precision():
    """Keep this targeted build in float32 arithmetic; restore caller settings afterwards."""
    old_cuda = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_cuda
        torch.backends.cudnn.allow_tf32 = old_cudnn


def inspect_song_source(directory: Path, required_stems: bool = False,
                        excerpt_start_s: float | None = None) -> tuple[dict, dict, dict]:
    """Verify real downloaded files and independently align vocals and accompaniment.

    The accompaniment is always an actual supplied source; it is never obtained
    by subtracting the vocal stem from the full mix. The inspection is saved and
    reviewed before a targeted replacement is rendered.
    """
    if excerpt_start_s is not None and (not math.isfinite(excerpt_start_s) or excerpt_start_s < 5):
        raise ValueError("explicit excerpt start must be finite and at least 5 seconds")
    safe_slug(directory.name)
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for key in ("title", "artist", "license", "license_url", "page", "files"):
        if not provenance.get(key):
            raise ValueError(f"{directory.name}: source provenance is missing {key}")
    license_url = urlparse(provenance["license_url"])
    if (license_url.hostname not in {"creativecommons.org", "www.creativecommons.org"}
            or not license_url.path.startswith("/licenses/by/")):
        raise ValueError("the public demo source must carry a verified CC BY license URL")
    roles = {role: [] for role in ("mix", "vocals", "accompaniment")}
    source_files = {}
    base = directory.resolve()
    for filename, metadata in provenance["files"].items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(base) or path == base or not path.is_file():
            raise ValueError(f"invalid or missing source path: {filename}")
        role = metadata.get("role")
        if role not in roles:
            raise ValueError(f"unsupported source role: {role}")
        digest = file_sha256(path)
        if metadata.get("sha256") != digest or not metadata.get("source"):
            raise ValueError(f"{filename}: source hash/URL missing or changed")
        audio = load_mono(path)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"{filename}: source audio is empty or non-finite")
        roles[role].append(audio)
        info = sf.info(str(path))
        source_files[filename] = {"role": role, "sha256": digest, "source": metadata["source"],
                                  "nickname": metadata.get("nickname", ""), "decoded_samples": len(audio),
                                  "decoded_duration_s": len(audio) / SR, "original_sample_rate": info.samplerate,
                                  "original_channels": info.channels, "original_frames": info.frames}
    if len(roles["mix"]) != 1 or not roles["vocals"] or (required_stems and not roles["accompaniment"]):
        raise ValueError("source requires exactly one full mix, vocal stem(s), and supplied accompaniment when required")
    mix = roles["mix"][0]
    if len(mix) < (5 + EXCERPT_S) * SR:
        raise ValueError("the song is too short for the 24-second excerpt with five seconds of preceding context")
    if excerpt_start_s is not None and excerpt_start_s > len(mix) / SR - EXCERPT_S:
        raise ValueError("explicit excerpt must fit completely within the full mix")
    aligned, alignment = {}, {}
    for role in ("vocals", "accompaniment"):
        signals = roles[role]
        if not signals:
            aligned[role] = None
            continue
        common = min(len(x) for x in signals)
        lag, correlation = best_lag(mix, sum(x[:common] for x in signals))
        value = sum(shift(x, lag, len(mix)) for x in signals).astype(np.float32)
        aligned[role] = value
        alignment[role] = {"lag_samples": lag, "lag_ms": lag / SR * 1000, "correlation": correlation,
                           "source_duration_s": [len(x) / SR for x in signals],
                           "coverage_start_s": max(0, lag) / SR,
                           "coverage_end_s": min(len(mix), max(len(x) + lag for x in signals)) / SR,
                           "local_probes": local_alignment_probes(mix, value)}
    if excerpt_start_s is None:
        start = pick_excerpt(aligned["vocals"])
        excerpt_selection = {"mode": "max_reference_vocal_energy"}
        excerpt_protocol = "maximum reference-vocal energy in a 24 s interval at integer-second starts, after 5 s"
    else:
        start = int(round(excerpt_start_s * SR))
        excerpt_selection = {"mode": "audited_explicit", "requested_start_s": float(excerpt_start_s)}
        excerpt_protocol = "explicit audited reference-vocal interval, rounded to nearest sample; at least 5 s preceding context"
    end = start + int(round(EXCERPT_S * SR))
    if start < 5 * SR or end > len(mix):
        raise ValueError("the selected excerpt must retain five seconds of context and fit within the mix")
    for role, value in aligned.items():
        if value is not None:
            a = alignment[role]
            if a["coverage_start_s"] > start / SR or a["coverage_end_s"] < end / SR:
                raise ValueError(f"{role}: downloaded sources do not cover the selected excerpt")
            if np.mean(value[start:end].astype(np.float64) ** 2) <= 1e-12:
                raise ValueError(f"{role}: selected excerpt has no usable energy")
    report = {"schema_version": 1, "slug": directory.name, "sample_rate": SR,
              "builder_sha256": file_sha256(Path(__file__)),
              "source_provenance_sha256": file_sha256(provenance_path), "sources": source_files,
              "attribution": {key: provenance.get(key, "") for key in
                              ("title", "artist", "artist_real_name", "license", "license_url", "page")},
              "mix_duration_s": len(mix) / SR, "alignment": alignment,
              "lag_protocol": "independent component-to-mix cross-correlation; first 90 s; decimation 4; search +/-5 s",
              "probe_protocol": "30 s early/middle/late aligned windows; residual search +/-0.1 s; skip below -60 dBFS",
              "excerpt_start_s": start / SR, "excerpt_s": EXCERPT_S,
              "excerpt_selection": excerpt_selection, "excerpt_protocol": excerpt_protocol}
    return provenance, {"mix": mix, "vocals": aligned["vocals"], "accompaniment": aligned["accompaniment"]}, report


def inspect_reviewed_song_source(directory: Path, reviewed: dict) -> tuple[dict, dict, dict]:
    """Replay the saved reference-only excerpt choice before accepting its source inspection."""
    selection = reviewed.get("excerpt_selection", {})
    mode = selection.get("mode")
    if mode == "max_reference_vocal_energy" and set(selection) == {"mode"}:
        requested_start = None
    elif mode == "audited_explicit" and set(selection) == {"mode", "requested_start_s"}:
        requested_start = selection["requested_start_s"]
        if not isinstance(requested_start, (int, float)) or isinstance(requested_start, bool):
            raise ValueError("reviewed explicit excerpt requires a numeric requested start")
    else:
        raise ValueError("reviewed inspection has an unknown or incomplete excerpt-selection rule")
    result = inspect_song_source(directory, required_stems=True, excerpt_start_s=requested_start)
    if not inspection_matches(reviewed, result[2]):
        raise ValueError("sources, alignment or excerpt differ from the reviewed inspection report")
    return result


def load_offline_reference(extra_path: str | None, device):
    if not extra_path:
        return None
    sys.path.insert(0, extra_path)
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    return get_model("htdemucs").eval().to(device), apply_model


def offline_vocals(reference, mix: np.ndarray, device) -> np.ndarray:
    htd, apply_model = reference
    wav = torch.from_numpy(np.stack([mix, mix]))
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std()
    if not torch.isfinite(sd) or sd <= 0:
        raise ValueError("full mix has no usable variation for the offline reference")
    torch.manual_seed(0)
    with torch.no_grad():
        output = apply_model(htd, ((wav - mu) / sd)[None].to(device), device=device,
                             shifts=1, split=True, overlap=0.25, progress=False)[0]
    return (output[htd.sources.index("vocals")] * sd.to(output.device) + mu.to(output.device)).mean(0).cpu().numpy()


def render_song(directory: Path, destination: Path, provenance: dict, audio: dict, inspection: dict,
                models: dict, reference, device, require_complete: bool = False) -> dict:
    """Process each complete song continuously, then export the same 24 s for all versions."""
    if require_complete and (any(models.get(config) is None for config, _, _ in MODELS) or reference is None):
        raise FileNotFoundError("replacement requires all A–E checkpoints and the original HTDemucs reference")
    mix, vocal, accompaniment = audio["mix"], audio["vocals"], audio["accompaniment"]
    start = int(round(inspection["excerpt_start_s"] * SR))
    sl = slice(start, start + int(round(EXCERPT_S * SR)))
    versions = [{"id": "mix", "label": "Mixture (input)", "kind": "input"}]
    write_mp3(destination / "mix.mp3", mix[sl])
    for config, label, latency in MODELS:
        if models[config] is None:
            continue
        model, stft = models[config]
        with torch.no_grad():
            estimate = separate(model, stft, mix)
        write_mp3(destination / f"{config}.mp3", estimate[sl])
        versions.append({"id": config, "label": label, "latency_ms": latency, "kind": "causal"})
    if reference is not None:
        write_mp3(destination / "htdemucs.mp3", offline_vocals(reference, mix, device)[sl])
        versions.append({"id": "htdemucs", "label": "HTDemucs · offline reference", "kind": "offline"})
    write_mp3(destination / "vocal_stem.mp3", vocal[sl])
    versions.append({"id": "vocal_stem", "label": "Original vocal stem", "kind": "reference"})
    song = {"slug": directory.name, **{key: provenance[key] for key in ("title", "artist", "license", "license_url", "page")},
            "artist_real_name": provenance.get("artist_real_name", ""),
            "featuring": featuring_of(directory.name, provenance),
            "excerpt_start_s": inspection["excerpt_start_s"], "excerpt_s": EXCERPT_S,
            "vocal_lag_ms": round(inspection["alignment"]["vocals"]["lag_ms"], 1),
            "source_provenance_sha256": inspection["source_provenance_sha256"], "versions": versions}
    if accompaniment is not None:
        gain = 10 ** (BVR_DB / 20) * np.sqrt(np.mean(vocal[sl].astype(np.float64) ** 2) /
                                            (np.mean(accompaniment[sl].astype(np.float64) ** 2) + 1e-12))
        bleed_input = (vocal + gain * accompaniment).astype(np.float32)
        write_mp3(destination / "bleed_input.mp3", bleed_input[sl])
        bleed = [{"id": "bleed_input", "label": f"Vocal mic with bleed ({BVR_DB:+.0f} dB)".replace("-", "−"), "kind": "input"}]
        for config, label, latency in MODELS:
            if models[config] is None:
                continue
            model, stft = models[config]
            with torch.no_grad():
                estimate = separate(model, stft, bleed_input)
            write_mp3(destination / f"bleed_{config}.mp3", estimate[sl])
            bleed.append({"id": f"bleed_{config}", "label": label, "latency_ms": latency, "kind": "causal"})
        bleed.append({"id": "vocal_stem", "label": "Original vocal stem", "kind": "reference"})
        song.update({"bleed": bleed, "accompaniment_lag_ms": round(inspection["alignment"]["accompaniment"]["lag_ms"], 1)})
    return song


def validate_replacement_assets(song: dict, directory: Path):
    safe_slug(song["slug"])
    expected_full = {"mix", "htdemucs", "vocal_stem", *BLEED_MODELS}
    expected_bleed = {"bleed_input", "vocal_stem", *(f"bleed_{config}" for config in BLEED_MODELS)}
    for key, expected in (("versions", expected_full), ("bleed", expected_bleed)):
        ids = [entry["id"] for entry in song.get(key, [])]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError(f"replacement {key} must contain every required version exactly once")
    if song.get("excerpt_s") != EXCERPT_S or not song.get("source_provenance_sha256"):
        raise ValueError("replacement excerpt/provenance is incomplete")
    for version in expected_full | expected_bleed:
        if not asset_complete(directory / f"{version}.mp3", EXCERPT_S):
            raise ValueError(f"replacement has an incomplete MP3/PNG pair: {version}")
    provenance_path = directory / "build_provenance.json"
    if not provenance_path.is_file():
        raise ValueError("replacement must retain its inspected sources and model provenance")
    build = json.loads(provenance_path.read_text())
    source = build.get("source_inspection", {})
    if (source.get("slug") != song["slug"] or source.get("sample_rate") != SR
            or source.get("source_provenance_sha256") != song["source_provenance_sha256"]
            or source.get("excerpt_s") != song["excerpt_s"] or source.get("excerpt_start_s") != song.get("excerpt_start_s")
            or any(song.get(key, "") != value for key, value in source.get("attribution", {}).items())
            or {item.get("role") for item in source.get("sources", {}).values()} != {"mix", "vocals", "accompaniment"}):
        raise ValueError("replacement identity, attribution or excerpt differs from inspected source provenance")
    checkpoints = build.get("model_checkpoints", {})
    if set(checkpoints) != set(BLEED_MODELS) or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                                                                for value in checkpoints.values()):
        raise ValueError("replacement requires checkpoint hashes for all A–E models")
    numerics = build.get("numerics", {})
    if (numerics.get("cuda_matmul_allow_tf32") is not False or numerics.get("cudnn_allow_tf32") is not False
            or numerics.get("dtype") != "float32"):
        raise ValueError("targeted replacement must record float32 inference with both TF32 flags disabled")
    expected_files = {f"{version}{suffix}" for version in expected_full | expected_bleed for suffix in (".mp3", ".png")}
    if set(build.get("asset_sha256", {})) != expected_files or any(
            file_sha256(directory / name) != digest for name, digest in build.get("asset_sha256", {}).items()):
        raise ValueError("replacement asset hashes are missing or differ from completed staged files")


def publish_replacement(manifest_path: Path, out_audio: Path, old_slug: str, song: dict,
                        staged_directory: Path, expected_manifest: bytes) -> dict:
    """Publish only complete new assets; roll them back if the atomic manifest swap fails."""
    safe_slug(old_slug)
    validate_replacement_assets(song, staged_directory)
    manifest = json.loads(expected_manifest)
    indexes = [i for i, old in enumerate(manifest["songs"]) if old["slug"] == old_slug]
    if len(indexes) != 1 or song["slug"] == old_slug or any(old["slug"] == song["slug"] for old in manifest["songs"]):
        raise ValueError("replacement requires one old slot and a distinct new song identity")
    if manifest.get("sample_rate") != SR:
        raise ValueError("existing manifest sample rate differs")
    if manifest_path.read_bytes() != expected_manifest:
        raise ValueError("demo manifest changed during rendering; refusing to overwrite concurrent changes")
    updated = copy.deepcopy(manifest)
    updated["songs"][indexes[0]] = song
    final_directory = out_audio / song["slug"]
    if final_directory.exists():
        raise FileExistsError(f"new song destination already exists: {final_directory}")
    out_audio.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(f".demo-replace-{os.getpid()}.json.tmp")
    published = False
    try:
        temporary_manifest.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n")
        staged_directory.replace(final_directory)
        published = True
        if manifest_path.read_bytes() != expected_manifest:
            raise ValueError("demo manifest changed before publication")
        temporary_manifest.replace(manifest_path)
    except BaseException:
        if published:
            final_directory.replace(staged_directory)
        raise
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return updated


@replacement_precision()
def replace_song(src: Path, old_slug: str, new_slug: str, device, extra_path: str,
                 alignment_report: Path, manifest_path: Path | None = None, out_audio: Path | None = None):
    safe_slug(old_slug)
    safe_slug(new_slug)
    manifest_path = manifest_path or REPO / "docs" / "demo.json"
    out_audio = out_audio or REPO / "docs" / "audio"
    original = manifest_path.read_bytes()
    current = json.loads(original)
    if sum(song["slug"] == old_slug for song in current["songs"]) != 1:
        raise ValueError("the requested old song does not identify exactly one existing slot")
    if any(song["slug"] == new_slug for song in current["songs"]) or (out_audio / new_slug).exists():
        raise ValueError("the replacement song must have a new, unused slug")
    directory = src / new_slug
    provenance, audio, inspection = inspect_reviewed_song_source(directory, json.loads(alignment_report.read_text()))
    models = {config: load_model(config, device) for config, _, _ in MODELS}
    if any(model is None for model in models.values()):
        raise FileNotFoundError("replacement requires every A–E seed-0 best checkpoint")
    reference = load_offline_reference(extra_path, device)
    if reference is None:
        raise FileNotFoundError("replacement requires the HTDemucs offline reference")
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".demo-replacement-", dir=out_audio.parent) as stage:
        staged_directory = Path(stage) / new_slug
        staged_directory.mkdir()
        song = render_song(directory, staged_directory, provenance, audio, inspection, models, reference, device, True)
        build = {"source_inspection": inspection, "replaced_slug": old_slug,
                 "previous_manifest_sha256": hashlib.sha256(original).hexdigest(),
                 "model_checkpoints": {config: file_sha256(REPO / "runs" / f"{config}_seed0" / "best.pt")
                                       for config, _, _ in MODELS},
                 "offline_reference": {"name": "htdemucs", "shifts": 1, "split": True, "overlap": 0.25, "seed": 0},
                 "numerics": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "dtype": "float32", "device": str(device), "torch_version": str(torch.__version__)},
                 "render_protocol": "full continuous mono44.1k inference; common24s excerpt; RMS-20dBFS/peak-1dBFS clips"}
        build["asset_sha256"] = {path.name: file_sha256(path) for path in sorted(staged_directory.iterdir()) if path.is_file()}
        (staged_directory / "build_provenance.json").write_text(json.dumps(build, indent=2, ensure_ascii=False) + "\n")
        updated = publish_replacement(manifest_path, out_audio, old_slug, song, staged_directory, original)
    print(f"Replaced {old_slug} with {new_slug} in its original slot; preserved every other song and model-table entry", flush=True)
    return updated


def build_bleed_only(src: Path, device, skip_existing: bool = False, model_names=None,
                     manifest_path: Path | None = None, out_audio: Path | None = None):
    """Extend existing bleed scenarios without re-separating full mixes or running HTDemucs.

    Song metadata, full-mix versions and original provenance are preserved. Alignment is
    checked against the original build so added versions use the same excerpt and reference.
    """
    manifest_path = manifest_path or REPO / "docs" / "demo.json"
    out_audio = out_audio or REPO / "docs" / "audio"
    manifest = json.loads(manifest_path.read_text())
    if manifest["sample_rate"] != SR:
        raise ValueError("existing demo sample rate differs; use a full rebuild")
    specs = {c: (label, lat) for c, label, lat in MODELS}
    names = list(BLEED_MODELS if model_names is None else model_names)
    unknown = set(names) - set(specs)
    if unknown:
        raise ValueError(f"unknown bleed models: {sorted(unknown)}")
    models = {}
    updated = 0
    for song in manifest["songs"]:
        d = src / song["slug"]
        prov = json.loads((d / "provenance.json").read_text())
        files = prov["files"]
        acc_files = [f for f, value in files.items() if value["role"] == "accompaniment"]
        if not acc_files:
            print(f"{d.name}: no accompaniment; preserving existing scenarios", flush=True)
            continue
        seconds = float(song["excerpt_s"])
        if abs(seconds - EXCERPT_S) > 1e-6:
            raise ValueError(f"{d.name}: excerpt length differs from the original build protocol")
        mix = load_mono(d / next(f for f, value in files.items() if value["role"] == "mix"))
        n = len(mix)
        vox_raw = [load_mono(d / f) for f, value in files.items() if value["role"] == "vocals"]
        lag, _ = best_lag(mix, sum(v[:min(len(v) for v in vox_raw)] for v in vox_raw))
        if round(lag / SR * 1000, 1) != song["vocal_lag_ms"]:
            raise ValueError(f"{d.name}: source alignment changed; use a full rebuild")
        voc = sum(shift(v, lag, n) for v in vox_raw)
        acc_raw = [load_mono(d / f) for f in acc_files]
        acc_lag = lag
        if "accompaniment_lag_ms" in song:
            acc_lag, _ = best_lag(mix, sum(a[:min(len(x) for x in acc_raw)] for a in acc_raw))
            if round(acc_lag / SR * 1000, 1) != song["accompaniment_lag_ms"]:
                raise ValueError(f"{d.name}: accompaniment alignment changed; use a full rebuild")
        acc = sum(shift(a, acc_lag, n) for a in acc_raw)
        s0 = int(round(float(song["excerpt_start_s"]) * SR))
        sl = slice(s0, s0 + int(round(seconds * SR)))
        if sl.start < 0 or sl.stop > n:
            raise ValueError(f"{d.name}: saved excerpt is outside the source audio")
        gain = 10 ** (BVR_DB / 20) * np.sqrt(np.mean(voc[sl] ** 2) / (np.mean(acc[sl] ** 2) + 1e-12))
        bleed_in = (voc + gain * acc).astype(np.float32)
        for version, audio in (("bleed_input", bleed_in[sl]), ("vocal_stem", voc[sl])):
            path = out_audio / d.name / f"{version}.mp3"
            if not skip_existing or not asset_complete(path, seconds):
                write_mp3(path, audio)
        entries = list(song.get("bleed", []))
        if not any(v["id"] == "bleed_input" for v in entries):
            entries.insert(0, {"id": "bleed_input", "label": f"Vocal mic with bleed ({BVR_DB:+.0f} dB)".replace("-", "−"),
                               "kind": "input"})
        if not any(v["id"] == "vocal_stem" for v in entries):
            entries.append({"id": "vocal_stem", "label": "Original vocal stem", "kind": "reference"})
        for cfg in names:
            path = out_audio / d.name / f"bleed_{cfg}.mp3"
            if not skip_existing or not asset_complete(path, seconds):
                if cfg not in models:
                    models[cfg] = load_model(cfg, device)
                if models[cfg] is None:
                    raise FileNotFoundError(f"requested bleed model has no checkpoint: {cfg}")
                model, stft = models[cfg]
                with torch.no_grad():
                    est = separate(model, stft, bleed_in)
                write_mp3(path, est[sl])
                print(f"{d.name}: generated bleed_{cfg}", flush=True)
            else:
                print(f"{d.name}: reuse bleed_{cfg}", flush=True)
            if not any(v["id"] == f"bleed_{cfg}" for v in entries):
                label, latency = specs[cfg]
                ref_index = next(i for i, v in enumerate(entries) if v["id"] == "vocal_stem")
                entries.insert(ref_index, {"id": f"bleed_{cfg}", "label": label,
                                           "latency_ms": latency, "kind": "causal"})
        song["bleed"] = entries
        updated += 1
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    temp.replace(manifest_path)
    print(f"updated bleed scenarios: {updated} songs; preserved full-mix versions", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/demo_ccmixter")
    ap.add_argument("--extra-path", help="directory containing the demucs package (for the HTDemucs reference)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--table-only", action="store_true", help="only refresh the model table in docs/demo.json")
    mode.add_argument("--bleed-only", action="store_true", help="extend bleed versions while preserving the existing full mixes")
    mode.add_argument("--inspect-song", metavar="SLUG", help="inspect source hashes, independent alignment and excerpt without rendering")
    mode.add_argument("--replace-song", metavar="OLD_SLUG", help="replace exactly one existing slot after staging a complete new song")
    ap.add_argument("--song", metavar="NEW_SLUG", help="new source slug for --replace-song")
    ap.add_argument("--required-stems", action="store_true", help="require supplied full mix, vocals and accompaniment, plus all model versions")
    ap.add_argument("--alignment-out", type=Path, help="save --inspect-song report for review")
    ap.add_argument("--alignment-report", type=Path, help="reviewed inspection report required by --replace-song")
    ap.add_argument("--excerpt-start", type=float, metavar="SECONDS", help="with --inspect-song, save an explicitly audited reference-vocal interval")
    ap.add_argument("--bleed-models", nargs="+", choices=BLEED_MODELS, default=BLEED_MODELS,
                    help="models to render for the bleed scenario (default: A–E)")
    ap.add_argument("--skip-existing", action="store_true", help="with --bleed-only, reuse valid existing MP3/PNG assets")
    args = ap.parse_args()
    if args.skip_existing and not args.bleed_only:
        ap.error("--skip-existing is supported with --bleed-only")
    if args.song and not args.replace_song:
        ap.error("--song is supported with --replace-song")
    if args.alignment_report and not args.replace_song:
        ap.error("--alignment-report is supported with --replace-song")
    if args.alignment_out and not args.inspect_song:
        ap.error("--alignment-out is supported with --inspect-song")
    if args.excerpt_start is not None and not args.inspect_song:
        ap.error("--excerpt-start is supported with --inspect-song; replacement replays its saved report")
    if args.inspect_song:
        _, _, inspection = inspect_song_source(Path(args.src) / safe_slug(args.inspect_song), args.required_stems,
                                              excerpt_start_s=args.excerpt_start)
        if args.alignment_out:
            args.alignment_out.parent.mkdir(parents=True, exist_ok=True)
            args.alignment_out.write_text(json.dumps(inspection, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(inspection, indent=2, ensure_ascii=False, allow_nan=False))
        return
    if args.replace_song:
        if not (args.song and args.required_stems and args.alignment_report and args.extra_path):
            ap.error("--replace-song requires --song, --required-stems, --alignment-report and --extra-path")
        if set(args.bleed_models) != set(BLEED_MODELS):
            ap.error("replacement requires all A–E bleed models")
        replace_song(Path(args.src), args.replace_song, args.song, torch.device(args.device), args.extra_path,
                     args.alignment_report)
        return
    if args.table_only:
        path = REPO / "docs" / "demo.json"
        manifest = json.loads(path.read_text())
        manifest["models"] = model_table()
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        copy_figures()
        print("refreshed the model table:", [m["id"] for m in manifest["models"]])
        return
    device = torch.device(args.device)
    if args.bleed_only:
        build_bleed_only(Path(args.src), device, args.skip_existing, args.bleed_models)
        return
    out_audio = REPO / "docs" / "audio"
    models = {c: load_model(c, device) for c, _, _ in MODELS}
    htd = None
    if args.extra_path:
        sys.path.insert(0, args.extra_path)
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        htd = get_model("htdemucs").eval().to(device)
    if args.required_stems and (any(model is None for model in models.values()) or htd is None):
        raise FileNotFoundError("required-stems build requires all A–E checkpoints and HTDemucs")
    songs = []
    for d in catalog_directories(Path(args.src)):
        if args.required_stems:
            inspect_song_source(d, required_stems=True)
        prov = json.loads((d / "provenance.json").read_text())
        files = prov["files"]
        mix = load_mono(d / next(f for f, v in files.items() if v["role"] == "mix"))
        n = len(mix)
        vox_raw = [load_mono(d / f) for f, v in files.items() if v["role"] == "vocals"]
        acc_raw = [load_mono(d / f) for f, v in files.items() if v["role"] == "accompaniment"]
        lag, peak = best_lag(mix, sum(v[:min(len(v) for v in vox_raw)] for v in vox_raw))
        voc = sum(shift(v, lag, n) for v in vox_raw)
        acc_lag, acc_peak = best_lag(mix, sum(a[:min(len(x) for x in acc_raw)] for a in acc_raw)) if acc_raw else (None, None)
        acc = sum(shift(a, acc_lag, n) for a in acc_raw) if acc_raw else None
        s0 = pick_excerpt(voc)
        sl = slice(s0, s0 + int(EXCERPT_S * SR))
        print(f"{d.name}: {n / SR:.0f} s, vocal lag {lag / SR * 1000:+.1f} ms (corr {peak:.2f}), excerpt at {s0 / SR:.0f} s",
              flush=True)
        versions = [{"id": "mix", "label": "Mixture (input)", "kind": "input"}]
        write_mp3(out_audio / d.name / "mix.mp3", mix[sl])
        for cfg, label, lat in MODELS:
            if models[cfg] is None:
                continue
            model, stft = models[cfg]
            with torch.no_grad():
                est = separate(model, stft, mix)
            write_mp3(out_audio / d.name / f"{cfg}.mp3", est[sl])
            versions.append({"id": cfg, "label": label, "latency_ms": lat, "kind": "causal"})
        if htd is not None:
            wav = torch.from_numpy(np.stack([mix, mix]))
            ref = wav.mean(0)
            mu, sd = ref.mean(), ref.std()
            with torch.no_grad():
                o = apply_model(htd, ((wav - mu) / sd)[None].to(device), device=device, shifts=1, split=True,
                                overlap=0.25, progress=False)[0]
            est = (o[htd.sources.index("vocals")] * sd.to(o.device) + mu.to(o.device)).mean(0).cpu().numpy()
            write_mp3(out_audio / d.name / "htdemucs.mp3", est[sl])
            versions.append({"id": "htdemucs", "label": "HTDemucs · offline reference", "kind": "offline"})
        write_mp3(out_audio / d.name / "vocal_stem.mp3", voc[sl])
        versions.append({"id": "vocal_stem", "label": "Original vocal stem", "kind": "reference"})
        song = {"slug": d.name, "title": prov["title"], "artist": prov["artist"],
                "artist_real_name": prov.get("artist_real_name", ""), "featuring": featuring_of(d.name, prov),
                "license": prov["license"],
                "license_url": prov["license_url"], "page": prov["page"], "excerpt_start_s": round(s0 / SR, 1),
                "excerpt_s": EXCERPT_S, "vocal_lag_ms": round(lag / SR * 1000, 1), "versions": versions}
        if acc is not None:
            song["accompaniment_lag_ms"] = round(acc_lag / SR * 1000, 1)
            g = 10 ** (BVR_DB / 20) * np.sqrt(np.mean(voc[sl] ** 2) / (np.mean(acc[sl] ** 2) + 1e-12))
            bleed_in = (voc + g * acc).astype(np.float32)
            write_mp3(out_audio / d.name / "bleed_input.mp3", bleed_in[sl])
            bv = [{"id": "bleed_input", "label": f"Vocal mic with bleed ({BVR_DB:+.0f} dB)".replace("-", "−"), "kind": "input"}]
            for cfg in args.bleed_models:
                if models[cfg] is None:
                    continue
                model, stft = models[cfg]
                with torch.no_grad():
                    est = separate(model, stft, bleed_in)
                write_mp3(out_audio / d.name / f"bleed_{cfg}.mp3", est[sl])
                bv.append({"id": f"bleed_{cfg}", "label": next(lab for c, lab, _ in MODELS if c == cfg),
                           "latency_ms": next(lat for c, _, lat in MODELS if c == cfg), "kind": "causal"})
            bv.append({"id": "vocal_stem", "label": "Original vocal stem", "kind": "reference"})
            song["bleed"] = bv
        songs.append(song)
    manifest = {"sample_rate": SR, "loudness": f"RMS {TARGET_RMS_DB:.0f} dBFS, peaks limited to {PEAK_DB:.0f} dBFS",
                "models": model_table(), "songs": songs}
    (REPO / "docs" / "demo.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    copy_figures()
    print("wrote docs/demo.json:", len(songs), "songs")


if __name__ == "__main__":
    main()
