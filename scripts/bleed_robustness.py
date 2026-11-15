"""Frozen-model robustness to three predeclared causal accompaniment paths.

Same MUSDB18-HQ excerpts as bleed_benchmark.py: 30 s scored, 5 s real context.
FIR prehistory is read before the context; delay is never aligned away. The clean
vocal stays dry, and the transformed accompaniment is normalized to -5 dB BVR
on the scored region. This is a synthetic feed-forward stress test, not feedback
simulation or a real-microphone evaluation. No training or parameter selection.

python -I scripts/bleed_robustness.py --device cuda --skip-existing
python -I scripts/bleed_robustness.py --summary-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

from dwmss.bleed_paths import energy, filtered_window, fixed_paths, scale_to_bvr  # noqa: E402
from dwmss.data import SR, STEMS, StemStore, list_tracks  # noqa: E402
from dwmss.metrics import si_sdr  # noqa: E402
from dwmss.model import MaskNet  # noqa: E402
from dwmss.stft import DualWindowSTFT  # noqa: E402
from scripts.bleed_benchmark import CONTEXT_S, EXCERPT_S, apply, mask_of, pick_excerpt  # noqa: E402

CONFIGS = ("B_sym256", "C_dual", "D_dual_2p9ms", "E_sym128")
DEFAULT_RUNS = [f"runs/{c}_seed{s}" for c in CONFIGS for s in (0, 1)]
GAMMAS = (0.5, 0.75, 1.0, 1.5, 2.0)
BVR_DB = -5.0
PAIRS = (("C_dual", "B_sym256"), ("D_dual_2p9ms", "E_sym128"))
METRICS = ("si_sdr", "vocal_loss_db", "bleed_atten_db")


def encoded(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(obj) -> str:
    return hashlib.sha256(encoded(obj).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as f:
            name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, obj):
    atomic_text(path, encoded(obj) + "\n")


def protocol(paths) -> dict:
    spec = {
        "schema_version": 1, "sample_rate": SR, "bvr_db": BVR_DB,
        "gammas": list(GAMMAS), "excerpt_s": EXCERPT_S, "context_s": CONTEXT_S,
        "required_seeds": [0, 1], "configs": list(CONFIGS),
        "paths": [p.provenance() for p in paths.values()],
        "excerpt_selection": "bleed_benchmark.pick_excerpt: highest vocal-energy 30 s on 1 s grid",
        "history": "Read FIR length-1 real samples before model context; keep path delay and original scene timing.",
        "normalization": "One gain on all transformed accompaniment stems; -5 dB relative to clean vocal on scored region.",
        "decomposition": "Mask inferred from each mixture; same frozen mask applied separately to clean vocal and transformed stems.",
        "equal_attenuation": "Seed-average metrics at each gamma first. Per song and pair, use midpoint of shared attainable attenuation range across both configs AND all three paths; no extrapolation. Report excluded songs and targets.",
        "inference": "Frozen best.pt checkpoints; no training, test tuning, or propagation-delay compensation.",
        "limitations": "Synthetic mono feed-forward FIRs, not measured rooms, real microphones or acoustic feedback. Bootstrap CIs resample songs, not training seeds.",
        "implementation_sha256": {str(p.relative_to(REPO)): file_sha256(p) for p in (
            Path(__file__), REPO / "dwmss/bleed_paths.py", REPO / "scripts/bleed_benchmark.py")},
    }
    return {**spec, "protocol_id": digest(spec)}


def excerpt_manifest(store, songs, paths) -> dict:
    history = max(len(p.impulse) - 1 for p in paths.values())
    length, context = round(EXCERPT_S * SR), round(CONTEXT_S * SR)
    excerpts = []
    for song in songs:
        n = store.lengths[song]
        score_start = pick_excerpt(store.read(song, "vocals", 0, n))
        context_start = score_start - context
        if context_start < 0 or score_start + length > n:
            raise ValueError(f"{song}: incomplete context or scored excerpt")
        files = {}
        for stem in STEMS:
            p = Path(store.dir) / song / f"{stem}.wav"
            stat = p.stat()
            files[stem] = {"path": str(p.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        excerpts.append({"song": song, "song_samples": n, "score_start_sample": score_start,
                         "score_end_sample": score_start + length, "context_start_sample": context_start,
                         "context_samples": context, "model_input_samples": context + length,
                         "path_read_start_sample": max(0, context_start - history),
                         "source_file_fingerprints": files})
    return {"songs": excerpts, "manifest_id": digest(excerpts),
            "source_identity_note": "Dataset WAVs identified by path/size/mtime, not content hashes; checkpoints and FIRs have SHA256 hashes."}


def valid_rows(rows, song, run_name, scenarios, config=None, seed=None) -> bool:
    if not isinstance(rows, list) or len(rows) != len(scenarios) * len(GAMMAS):
        return False
    expected = {(s, g) for s in scenarios for g in GAMMAS}
    try:
        if {(r["scenario"], float(r["gamma"])) for r in rows} != expected:
            return False
        return all(r["song"] == song and r["run"] == run_name and r["bvr_db"] == BVR_DB
                   and (config is None or r["config"] == config)
                   and (seed is None or r["seed"] == seed)
                   and all(np.isfinite(r[k]) for k in METRICS + ("si_sdr_input", "actual_bvr_db",))
                   and abs(r["actual_bvr_db"] - BVR_DB) < 1e-4 for r in rows)
    except (KeyError, TypeError, ValueError):
        return False


def read_cache(path, identity, song, run_name, scenarios, config=None, seed=None):
    try:
        cached = json.loads(path.read_text())
        rows = cached["rows"]
        if (cached["complete"] and cached["identity"] == identity
                and cached["rows_sha256"] == digest(rows)
                and valid_rows(rows, song, run_name, scenarios, config, seed)):
            return rows
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def db_ratio(numerator, denominator):
    return float(10 * np.log10((energy(numerator) + 1e-12) / (energy(denominator) + 1e-12)))


@torch.no_grad()
def song_rows(model, stft, store, excerpt, paths, run_meta, device):
    song = excerpt["song"]
    a0, n = excerpt["context_start_sample"], excerpt["model_input_samples"]
    p0, context = excerpt["path_read_start_sample"], excerpt["context_samples"]
    sc = slice(context, n)
    vocal = store.read(song, "vocals", a0, n)
    history = {s: store.read(song, s, p0, n + a0 - p0) for s in STEMS[1:]}
    V = stft.analysis(torch.as_tensor(vocal, device=device)[None])
    rows = []
    for name, path in paths.items():
        transformed = {s: filtered_window(x, path.impulse, a0 - p0, n) for s, x in history.items()}
        stems, gain, achieved = scale_to_bvr(vocal, transformed, sc, BVR_DB)
        acc = sum(stems.values())
        mix = (vocal + acc).astype(np.float32)
        S = {s: stft.analysis(torch.as_tensor(x, device=device)[None]) for s, x in stems.items()}
        Mm = mask_of(model, stft, mix, device)
        scenario_rows = []
        decomposition_error = None
        for gamma in GAMMAS:
            mask = Mm ** gamma
            out_v = apply(stft, mask, V, n)[sc]
            out_s = {s: apply(stft, mask, spec, n)[sc] for s, spec in S.items()}
            out_a = sum(out_s.values())
            if gamma == 1.0:
                direct_spec = stft.analysis(torch.as_tensor(mix, device=device)[None])
                direct = apply(stft, mask, direct_spec, n)[sc]
                decomposition_error = float(np.max(np.abs(direct - (out_v + out_a))))
                if not np.allclose(direct, out_v + out_a, atol=2e-5, rtol=2e-4):
                    raise AssertionError(f"frozen-mask decomposition failed for {run_meta['run']}/{song}/{name}")
            row = {"song": song, "run": run_meta["run"], "config": run_meta["config"],
                   "seed": run_meta["seed"], "scenario": name, "gamma": gamma,
                   "bvr_db": BVR_DB, "actual_bvr_db": achieved, "accompaniment_gain": gain,
                   "si_sdr": si_sdr(out_v + out_a, vocal[sc]), "si_sdr_input": si_sdr(mix[sc], vocal[sc]),
                   "vocal_preserved_si_sdr": si_sdr(out_v, vocal[sc]),
                   "vocal_loss_db": db_ratio(out_v, vocal[sc]),
                   "bleed_atten_db": db_ratio(acc[sc], out_a)}
            row.update({f"atten_{s}_db": db_ratio(stems[s][sc], out_s[s]) for s in stems})
            scenario_rows.append(row)
        for row in scenario_rows:
            row["gamma1_decomposition_max_abs_error"] = decomposition_error
        rows.extend(scenario_rows)
    return rows


def run_one(run, store, excerpts, paths, spec, out, device, skip_existing):
    checkpoint = run / "best.pt"
    ck_hash = file_sha256(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg["name"] not in CONFIGS or int(ck["seed"]) not in (0, 1):
        raise ValueError("this frozen protocol requires B/C/D/E checkpoints with seeds 0 or 1")
    meta = {"run": run.name, "config": cfg["name"], "seed": int(ck["seed"]),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": ck_hash,
            "checkpoint_step": int(ck["step"]), "model_config": cfg["model"], "stft": cfg["stft"],
            "protocol_id": spec["protocol_id"], "excerpt_manifest_id": excerpts["manifest_id"],
            "device": str(device), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    experiment_id = digest({k: meta[k] for k in ("run", "config", "seed", "checkpoint_sha256",
                                                "protocol_id", "excerpt_manifest_id")})
    meta["experiment_id"] = experiment_id
    atomic_json(out / "run_provenance" / f"{run.name}.json", meta)
    stft = DualWindowSTFT(cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"]).to(device)
    model = MaskNet(**cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    del ck
    all_rows = []
    for i, excerpt in enumerate(excerpts["songs"]):
        song = excerpt["song"]
        cache = out / "cache" / run.name / f"{hashlib.sha256(song.encode()).hexdigest()[:20]}.json"
        identity = {"experiment_id": experiment_id, "song": song}
        rows = read_cache(cache, identity, song, run.name, paths, meta["config"], meta["seed"]) if skip_existing else None
        reused = rows is not None
        if rows is None:
            rows = song_rows(model, stft, store, excerpt, paths, meta, device)
            if not valid_rows(rows, song, run.name, paths, meta["config"], meta["seed"]):
                raise AssertionError(f"invalid result rows for {run.name}/{song}")
            atomic_json(cache, {"identity": identity, "complete": True, "rows": rows, "rows_sha256": digest(rows)})
        all_rows.extend(rows)
        print(f"{run.name} [{i + 1}/{len(excerpts['songs'])}] {song}{' (validated cache)' if reused else ''}", flush=True)
    csv = out / f"{run.name}.csv"
    atomic_text(csv, pd.DataFrame(all_rows).to_csv(index=False))
    atomic_json(out / "completions" / f"{run.name}.json",
                {"complete": True, "experiment_id": experiment_id, "csv_sha256": file_sha256(csv),
                 "songs": len(excerpts["songs"]), "rows": len(all_rows)})


def completed_frames(out, spec, excerpts, run_names):
    frames, missing = [], []
    expected_songs = {x["song"] for x in excerpts["songs"]}
    scenarios = [p["name"] for p in spec["paths"]]
    for name in run_names:
        try:
            meta = json.loads((out / "run_provenance" / f"{name}.json").read_text())
            done = json.loads((out / "completions" / f"{name}.json").read_text())
            csv = out / f"{name}.csv"
            if (not done["complete"] or done["experiment_id"] != meta["experiment_id"]
                    or meta["protocol_id"] != spec["protocol_id"]
                    or meta["excerpt_manifest_id"] != excerpts["manifest_id"]
                    or done["csv_sha256"] != file_sha256(csv)):
                raise ValueError("stale or inconsistent completion")
            frame = pd.read_csv(csv)
            if (set(frame.song) != expected_songs or len(frame) != done["rows"]
                    or done["songs"] != len(expected_songs)):
                raise ValueError("incomplete song set")
            if not all(valid_rows(g.to_dict("records"), song, name, scenarios, meta["config"], meta["seed"])
                       for song, g in frame.groupby("song")):
                raise ValueError("incomplete rows")
            frames.append(frame)
        except (OSError, KeyError, ValueError, TypeError):
            missing.append(name)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), missing


def bootstrap_difference(values, reps=10000):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("expected finite paired differences")
    samples = np.random.default_rng(0).choice(x, size=(reps, len(x)))
    med_ci = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    mean_ci = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return {"songs": len(x), "median": float(np.median(x)), "median_ci_low": float(med_ci[0]),
            "median_ci_high": float(med_ci[1]), "mean": float(x.mean()), "mean_ci_low": float(mean_ci[0]),
            "mean_ci_high": float(mean_ci[1]), "fraction_positive": float((x > 0).mean())}


def paired_stat(a, b, **labels):
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return {**labels, **bootstrap_difference(aligned.a - aligned.b)} if len(aligned) else None


def equal_attenuation(per_song, scenarios):
    """Same song/pair target across dry and both transformed paths; no extrapolation."""
    rows = []
    for dual, short in PAIRS:
        available = per_song[per_song.config.isin([dual, short])]
        for song, group in available.groupby("song"):
            curves = {(c, p): g.groupby("bleed_atten_db")["vocal_loss_db"].mean().sort_index()
                      for (c, p), g in group.groupby(["config", "scenario"])}
            if set(curves) != {(c, p) for c in (dual, short) for p in scenarios}:
                continue
            lo = max(float(g.index.min()) for g in curves.values())
            hi = min(float(g.index.max()) for g in curves.values())
            if hi <= lo:
                continue
            target = (lo + hi) / 2
            for (config, scenario), curve in curves.items():
                rows.append({"pair": f"{dual} - {short}", "song": song, "config": config,
                             "scenario": scenario, "target_bleed_atten_db": target,
                             "shared_range_low_db": lo, "shared_range_high_db": hi,
                             "vocal_loss_db": float(np.interp(target, curve.index, curve.values))})
    return pd.DataFrame(rows, columns=["pair", "song", "config", "scenario", "target_bleed_atten_db",
                                       "shared_range_low_db", "shared_range_high_db", "vocal_loss_db"])


def summarize(out, run_names):
    spec = json.loads((out / "protocol.json").read_text())
    excerpts = json.loads((out / "excerpts.json").read_text())
    scenarios = [p["name"] for p in spec["paths"]]
    raw, missing = completed_frames(out, spec, excerpts, run_names)
    lines = ["# Frozen-model robustness to synthetic accompaniment paths", "",
             "MUSDB18-HQ test songs; 30 s scored excerpts and 5 s unscored recurrent-state context. "
             "FIR support is read before the context. Original timing and path delays are retained. "
             "Accompaniment is re-normalized to -5 dB BVR after each path. No retraining or path tuning.", "",
             f"Protocol `{spec['protocol_id']}`. Frozen paths: dry identity, mild filter + delay, finite synthetic room FIR. "
             "Exact coefficients and path parameters: `protocol.json`; checkpoint SHA256 hashes: `run_provenance/`; "
             "scored excerpts and dataset-file fingerprints: `excerpts.json`.", "",
             "Synthetic feed-forward stress tests, not measured rooms, real microphones or acoustic feedback. "
             "All comparisons average seeds 0 and 1 per song first. Bootstrap 95% CIs resample songs "
             "(10,000 draws, fixed seed 0); they do not include training-seed uncertainty.", ""]
    if missing:
        lines += ["Incomplete or invalid runs (excluded): " + ", ".join(missing) + ".", ""]
    if raw.empty:
        atomic_text(out / "summary.md", "\n".join(lines + ["No complete runs available.", ""]))
        return
    complete_configs = [c for c, g in raw.groupby("config") if set(g.seed) == {0, 1}]
    incomplete_configs = sorted(set(raw.config) - set(complete_configs))
    if incomplete_configs:
        lines += ["Configs missing one of the two required seeds (excluded from summaries): "
                  + ", ".join(incomplete_configs) + ".", ""]
    raw = raw[raw.config.isin(complete_configs)]
    if raw.empty:
        atomic_text(out / "summary.md", "\n".join(lines + ["No config has both completed seeds.", ""]))
        return
    numeric = [c for c in raw.select_dtypes(include="number") if c not in ("seed", "gamma", "bvr_db")]
    per_song = raw.groupby(["config", "song", "scenario", "gamma"], as_index=False)[numeric].mean()
    atomic_text(out / "per_song.csv", per_song.to_csv(index=False))
    med = per_song.groupby(["config", "scenario", "gamma"], as_index=False)[numeric].median()
    atomic_text(out / "summary.csv", med.to_csv(index=False))
    lines += ["## Primary operating point (gamma = 1)", "",
              "Medians over songs after seed averaging. Vocal loss is output vocal energy relative to clean vocal "
              "(0 dB = no energy loss); bleed attenuation is positive when accompaniment energy is removed.", "",
              "| config | path | songs | SI-SDR (dB) | vocal loss (dB) | bleed attenuation (dB) |",
              "|---|---|---:|---:|---:|---:|"]
    main = per_song[per_song.gamma == 1.0]
    for (config, scenario), group in main.groupby(["config", "scenario"]):
        m = group[list(METRICS)].median()
        lines.append(f"| {config} | {scenario} | {len(group)} | {m.si_sdr:.2f} | {m.vocal_loss_db:+.2f} | {m.bleed_atten_db:.2f} |")
    contrasts = []
    for metric in METRICS:
        wide = main.pivot(index="song", columns=["config", "scenario"], values=metric)
        for dual, short in PAIRS:
            for scenario in scenarios:
                if (dual, scenario) in wide and (short, scenario) in wide:
                    stat = paired_stat(wide[dual, scenario], wide[short, scenario], kind="dual_minus_short",
                                       comparison=f"{dual} - {short}", scenario=scenario, metric=metric)
                    if stat:
                        contrasts.append(stat)
        for config in complete_configs:
            for scenario in scenarios:
                if scenario != "dry" and (config, scenario) in wide and (config, "dry") in wide:
                    stat = paired_stat(wide[config, scenario], wide[config, "dry"], kind="path_minus_dry",
                                       comparison=config, scenario=scenario, metric=metric)
                    if stat:
                        contrasts.append(stat)
    equal = equal_attenuation(per_song, scenarios)
    atomic_text(out / "equal_attenuation_per_song.csv", equal.to_csv(index=False))
    for dual, short in PAIRS:
        pair = f"{dual} - {short}"
        group = equal[equal.pair == pair]
        if group.empty:
            continue
        wide = group.pivot(index="song", columns=["config", "scenario"], values="vocal_loss_db")
        for scenario in scenarios:
            stat = paired_stat(wide[dual, scenario], wide[short, scenario], kind="equal_attenuation_dual_minus_short",
                               comparison=pair, scenario=scenario, metric="vocal_loss_db")
            contrasts.append(stat)
        for config in (dual, short):
            for scenario in scenarios:
                if scenario != "dry":
                    contrasts.append(paired_stat(wide[config, scenario], wide[config, "dry"],
                                                 kind="equal_attenuation_path_minus_dry", comparison=config,
                                                 scenario=scenario, metric="vocal_loss_db"))
    contrast = pd.DataFrame([c for c in contrasts if c])
    atomic_text(out / "paired_comparisons.csv", contrast.to_csv(index=False))
    for kind, title, metric in (("dual_minus_short", "Paired SI-SDR gain at gamma = 1", "si_sdr"),
                                ("path_minus_dry", "SI-SDR change from dry at gamma = 1", "si_sdr"),
                                ("equal_attenuation_dual_minus_short", "Paired vocal preservation at equal attainable attenuation", "vocal_loss_db"),
                                ("equal_attenuation_path_minus_dry", "Vocal-preservation change from dry at the same attenuation", "vocal_loss_db")):
        lines += ["", f"## {title}", ""]
        if kind.startswith("equal_"):
            lines += ["For each song and model pair, take the midpoint of the attenuation range shared by "
                      "both configs and all three paths, using seed-averaged five-point gamma curves. "
                      "Use that same target for dry and transformed paths. Linear interpolation only, no "
                      "extrapolation; songs without a shared range are excluded. Positive differences mean "
                      "more vocal energy is kept. This compares energy preservation, not perceptual quality.", ""]
        elif kind == "path_minus_dry":
            lines += ["Negative changes indicate degradation relative to the identity path on the same songs.", ""]
        lines += ["| comparison | path | paired songs | median difference (dB) [95% CI] | mean difference (dB) [95% CI] |",
                  "|---|---|---:|---|---|"]
        if contrast.empty:
            continue
        for _, r in contrast[(contrast.kind == kind) & (contrast.metric == metric)].iterrows():
            lines.append(f"| {r.comparison} | {r.scenario} | {int(r.songs)} | {r['median']:+.2f} "
                         f"[{r.median_ci_low:+.2f}, {r.median_ci_high:+.2f}] | {r['mean']:+.2f} "
                         f"[{r.mean_ci_low:+.2f}, {r.mean_ci_high:+.2f}] |")
    lines += ["", "## Equal-attenuation coverage", "",
              "| pair | eligible / total songs | median target attenuation (dB) | target range across songs (dB) |",
              "|---|---:|---:|---:|"]
    for dual, short in PAIRS:
        pair = f"{dual} - {short}"
        targets = equal[equal.pair == pair].drop_duplicates("song").target_bleed_atten_db
        total = len(excerpts["songs"])
        if len(targets):
            lines.append(f"| {pair} | {len(targets)} / {total} | {targets.median():.2f} | {targets.min():.2f} to {targets.max():.2f} |")
        else:
            lines.append(f"| {pair} | 0 / {total} | unavailable | unavailable |")
    lines += ["", "All per-song gamma curves: `per_song.csv`. All paired metrics (including bleed attenuation "
              "and vocal-loss changes at gamma = 1): `paired_comparisons.csv`. Equal-attenuation targets and "
              "interpolated scores: `equal_attenuation_per_song.csv`. Each per-song cache checks its protocol, "
              "checkpoint and excerpt identity; a run is complete only after every song and every path/gamma "
              "has been validated and the atomic CSV checksum recorded.", ""]
    atomic_text(out / "summary.md", "\n".join(lines))
    atomic_json(out / "summary_provenance.json", {"protocol_id": spec["protocol_id"],
                "excerpt_manifest_id": excerpts["manifest_id"], "included_configs": complete_configs,
                "incomplete_runs": missing, "songs": len(excerpts["songs"]), "bootstrap_seed": 0,
                "bootstrap_resamples": 10000})
    print(f"wrote {out / 'summary.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--max-songs", type=int)
    ap.add_argument("--device")
    ap.add_argument("--musdb-root", default="data/musdb18hq")
    ap.add_argument("--out", default=str(REPO / "results/bleed_robustness"))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="reuse only complete identity-validated per-song caches")
    args = ap.parse_args()
    if args.max_songs is not None and args.max_songs < 1:
        ap.error("--max-songs must be positive")
    out = Path(args.out)
    if args.max_songs is not None and out.resolve() == (REPO / "results/bleed_robustness").resolve():
        ap.error("--max-songs is a smoke test; choose a separate --out directory to protect the full-test results")
    out.mkdir(parents=True, exist_ok=True)
    run_paths = [p if p.is_absolute() else REPO / p for p in map(Path, args.runs)]
    if len({p.name for p in run_paths}) != len(run_paths):
        ap.error("run directory names must be unique")
    if args.summary_only:
        summarize(out, [p.name for p in run_paths])
        return
    for run in run_paths:
        if not (run / "best.pt").exists():
            raise FileNotFoundError(run / "best.pt")
    paths = fixed_paths(SR)
    spec = protocol(paths)
    songs = list_tracks(args.musdb_root, "test")
    if args.max_songs is None and len(songs) != 50:
        raise ValueError(f"full protocol expects 50 test songs, found {len(songs)}")
    songs = songs[:args.max_songs] if args.max_songs else songs
    store = StemStore(args.musdb_root, "test", songs)
    excerpts = excerpt_manifest(store, songs, paths)
    atomic_json(out / "protocol.json", spec)
    atomic_json(out / "excerpts.json", excerpts)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for run in run_paths:
        run_one(run, store, excerpts, paths, spec, out, device, args.skip_existing)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summarize(out, [p.name for p in run_paths])


if __name__ == "__main__":
    main()
