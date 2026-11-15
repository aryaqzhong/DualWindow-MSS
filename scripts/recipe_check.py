"""A bounded, symmetric learning-rate follow-up at 2.9 ms.

Freeze before training; train four new runs without test evaluation; select one
learning rate per architecture using the mean of two validation-selected scores;
only then evaluate selected checkpoints. Original results are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dwmss.config import load_config  # noqa: E402

OUT = REPO / "results" / "recipe_check"
SEEDS = (0, 1)
TIE_TOLERANCE = 1e-9
FAMILIES = {"wide_short": ("E_sym128_compute", "E_sym128_compute_lr3e4"),
            "dual": ("D_dual_2p9ms", "D_dual_2p9ms_lr3e4")}
SOURCE_FILES = ("scripts/recipe_check.py", "scripts/train.py", "dwmss/config.py", "dwmss/model.py",
                "dwmss/stft.py", "dwmss/data.py", "dwmss/losses.py", "dwmss/separator.py", "dwmss/metrics.py",
                "slurm/train_recipe_check.sbatch", "slurm/evaluate_recipe_check.sbatch")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(value)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame):
    atomic_text(path, frame.to_csv(index=False))


def validate_recipe_pair(base: dict, extra: dict):
    expected = copy.deepcopy(base)
    expected["name"] += "_lr3e4"
    expected["train"]["lr"] = 3e-4
    if extra != expected or not math.isclose(float(base["train"]["lr"]), 1e-3):
        raise ValueError("The extra candidate must change only name and learning rate 1e-3 to 3e-4")
    if base["train"]["max_steps"] != 200000 or base["stft"]["n_synthesis"] != 128:
        raise ValueError("This follow-up is restricted to the original 200000-step, 2.9 ms setting")


def validation_log(path: Path) -> tuple[list[dict], list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    training, validation = [], []
    for row in records:
        if "step" not in row or not isinstance(row["step"], int):
            raise ValueError(f"{path}: missing integer step")
        for value in row.values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{path}: non-finite log value")
        if "loss" in row:
            training.append(row)
        elif "val_si_sdr_median" in row:
            validation.append(row)
        else:
            raise ValueError(f"{path}: unrecognized log record")
    if [r["step"] for r in training] != list(range(100, 200001, 100)):
        raise ValueError(f"{path}: requires exactly one training record at every 100 steps through 200000")
    if [r["step"] for r in validation] != list(range(5000, 200001, 5000)):
        raise ValueError(f"{path}: requires exactly one validation record at every 5000 steps through 200000")
    best = -math.inf
    for row in validation:
        improved = row["val_si_sdr_median"] > best
        best = max(best, row["val_si_sdr_median"])
        if not math.isclose(row["best"], best, abs_tol=1e-9) or row["improved"] != improved:
            raise ValueError(f"{path}: running validation best is inconsistent")
    return training, validation


def verify_run(config: str, seed: int, repo: Path = REPO) -> dict:
    import torch
    from dwmss.data import VALIDATION_TRACKS

    run = repo / "runs" / f"{config}_seed{seed}"
    cfg = load_config(str(repo / "configs" / f"{config}.yaml"))
    if (run / "DONE").read_text().strip() != "finished 200000 steps":
        raise ValueError(f"{run}: 200000-step completion marker required")
    meta = json.loads((run / "meta.json").read_text())
    if (meta.get("config") != cfg or meta.get("seed") != seed or len(meta.get("valid_tracks", [])) != 14
            or set(meta["valid_tracks"]) != set(VALIDATION_TRACKS)):
        raise ValueError(f"{run}: metadata differs from the prescribed configuration/seed/validation split")
    _, validation = validation_log(run / "log.jsonl")
    best = max(validation, key=lambda row: row["val_si_sdr_median"])
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if (checkpoint.get("config") != cfg or checkpoint.get("seed") != seed
            or checkpoint.get("step") != best["step"] or not checkpoint.get("model")
            or not math.isclose(checkpoint.get("best", -math.inf), best["val_si_sdr_median"], abs_tol=1e-9)):
        raise ValueError(f"{run}: best.pt does not match the first maximum validation score")
    del checkpoint
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    if last.get("step") != 200000 or last.get("config") != cfg or last.get("seed") != seed:
        raise ValueError(f"{run}: final checkpoint did not complete the prescribed training")
    del last
    return {"config": config, "seed": seed, "run": str(run.resolve()), "best_step": best["step"],
            "best_validation_median": float(best["val_si_sdr_median"]),
            "hashes": {name: sha256(run / name) for name in ("meta.json", "log.jsonl", "best.pt", "last.pt", "DONE")}}


def validate_evaluation(frame: pd.DataFrame, metadata: dict, identity: dict, expected_tracks: list[str]):
    needed = {"track", "seconds", "si_sdr", "usdr", "si_sdri", "si_sdr_mixture"}
    if not needed.issubset(frame.columns) or len(frame) != 50 or frame.track.nunique() != 50:
        raise ValueError("Evaluation requires exactly 50 unique full-song records and all primary metrics")
    if set(frame.track) != set(expected_tracks) or not np.isfinite(frame[list(needed - {"track"})].to_numpy()).all():
        raise ValueError("Evaluation tracks differ or scores are non-finite")
    if (metadata.get("config") != identity["config"] or metadata.get("step") != identity["best_step"]
            or metadata.get("n_tracks") != 50 or metadata.get("ckpt") != "best.pt" or metadata.get("subset") != "test"):
        raise ValueError("Evaluation metadata does not identify the validation-selected full test evaluation")
    if not np.allclose(frame.si_sdri, frame.si_sdr - frame.si_sdr_mixture, atol=1e-7, rtol=0):
        raise ValueError("SI-SDR improvement does not match the stored mixture baseline")


def original_evaluation(identity: dict, tracks: list[str]) -> tuple[pd.DataFrame, dict]:
    stem = f"{identity['config']}_seed{identity['seed']}_test"
    path = REPO / "results" / "eval" / f"{stem}.csv"
    metadata_path = path.with_suffix(".json")
    frame, metadata = pd.read_csv(path), json.loads(metadata_path.read_text())
    validate_evaluation(frame, metadata, identity, tracks)
    return frame, {"csv_path": str(path), "csv_sha256": sha256(path),
                   "json_path": str(metadata_path), "json_sha256": sha256(metadata_path)}


def make_design() -> dict:
    import yaml
    from dwmss.data import STEMS, list_tracks

    resolved, config_files = {}, set()
    originals, evaluations = {}, {}
    for base, extra in FAMILIES.values():
        configs = [load_config(str(REPO / "configs" / f"{name}.yaml")) for name in (base, extra)]
        validate_recipe_pair(*configs)
        for name, cfg in zip((base, extra), configs):
            resolved[name] = cfg
            path = REPO / "configs" / f"{name}.yaml"
            while True:
                config_files.add(str(path.relative_to(REPO)))
                parent = (yaml.safe_load(path.read_text()) or {}).get("inherit")
                if not parent:
                    break
                path = path.parent / parent
    root = Path(next(iter(resolved.values()))["data"]["musdb_root"])
    tracks = list_tracks(str(root), "test")
    if len(tracks) != 50:
        raise ValueError("Expected all 50 MUSDB18-HQ test songs")
    sources = {}
    for track in tracks:
        for stem in STEMS:
            p = root / "test" / track / f"{stem}.wav"
            st = p.stat()
            sources[str(p)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    for base, _ in FAMILIES.values():
        for seed in SEEDS:
            name = f"{base}_seed{seed}"
            originals[name] = verify_run(base, seed)
            _, evaluations[name] = original_evaluation(originals[name], tracks)
    return {"schema_version": 1, "scope": "Exploratory 2.9 ms follow-up motivated by already observed test results",
            "families": FAMILIES, "seeds": list(SEEDS), "resolved_configs": resolved,
            "selection": "Per architecture: arithmetic mean of each seed's best validation median; ties within 1e-9 dB favor original LR",
            "selection_tie_tolerance_db": TIE_TOLERANCE,
            "candidate_learning_rates": [0.001, 0.0003], "steps_per_run": 200000,
            "file_hashes": {name: sha256(REPO / name) for name in sorted(set(SOURCE_FILES) | config_files)},
            "original_runs": originals, "original_evaluations": evaluations,
            "test_tracks": tracks, "dataset_fingerprints": sources,
            "dataset_fingerprint_scope": "absolute paths, file sizes and nanosecond mtimes; not audio-content hashes"}


def freeze(out: Path):
    path = out / "design.json"
    if not path.exists():
        for _, extra in FAMILIES.values():
            for seed in SEEDS:
                run = REPO / "runs" / f"{extra}_seed{seed}"
                if any((run / name).exists() for name in ("log.jsonl", "best.pt", "last.pt")):
                    raise ValueError(f"{run}: candidate training has already started; refuse retroactive design freeze")
    design = make_design()
    # JSON canonicalizes tuple values so subsequent comparison is exact.
    design = json.loads(json.dumps(design))
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError("A different recipe-check design is already frozen; refuse to overwrite it")
    if not path.exists():
        atomic_json(path, design)
    lines = ["# Bounded learning-rate check at 2.9 ms", "",
             "Design fixed before new training; summary.md contains completed results when available.", "",
             "This exploratory follow-up was motivated by the original test results and validation diagnostics. "
             "It is not an independent confirmation on a new test set.", "",
             "Two architectures: short-window GRU-800 (E_sym128_compute) and dual-window GRU-256 (D_dual_2p9ms). "
             "Both receive one identical extra learning-rate candidate: 0.0003, alongside the existing 0.001. "
             "Weight decay stays 0.01; all other settings, data, 200000 updates, seeds 0/1, "
             "validation-checkpoint rule and schedule stay fixed. No architecture changes.", "",
             "Choose one learning rate per architecture by the arithmetic mean across the two seeds of each seed's "
             "own best validation median SI-SDR. Ties within 1e-9 dB favor the original learning rate. The choices and "
             "all input hashes are written atomically to selection.json before new test evaluation. "
             "Only the selected setting is evaluated on all 50 full test songs. "
             "Already selected original settings reuse their frozen evaluation after metadata checks.", "",
             "Each selected setting averages seeds within songs before song medians and paired comparisons. "
             "Bootstrap CIs resample 50 songs 10000 times with seed 0; they do not include training-seed uncertainty. "
             "This one-learning-rate check does not establish optimized architectures or general computational optimality.", "",
             "design.json freezes resolved configurations, all inherited configuration files, source code, "
             "original logs/checkpoints/evaluations and dataset path/size/mtime fingerprints. "
             "New runs and outputs are separate from the original MAC-matched experiment."]
    atomic_text(out / "design.md", "\n".join(lines) + "\n")
    print(f"Frozen design: {path}", flush=True)


def check_design(out: Path) -> dict:
    design = json.loads((out / "design.json").read_text())
    current = json.loads(json.dumps(make_design()))
    if design != current:
        raise ValueError("Recipe-check design differs: configuration, code, original artifacts or dataset changed")
    return design


def choose_settings(scores: dict[str, dict[str, list[float]]]) -> dict[str, str]:
    choices = {}
    if set(scores) != set(FAMILIES):
        raise ValueError("Both architectures must have identical two-setting validation budgets")
    for family, (base, extra) in FAMILIES.items():
        candidates = scores[family]
        if set(candidates) != {base, extra}:
            raise ValueError("Both predefined learning-rate settings are required")
        for values in candidates.values():
            if len(values) != 2 or not np.isfinite(values).all():
                raise ValueError("Selection requires two finite per-seed validation maxima per candidate")
        choices[family] = extra if np.mean(candidates[extra]) > np.mean(candidates[base]) + TIE_TOLERANCE else base
    return choices


def select(out: Path) -> dict:
    check_design(out)
    identities, scores = {}, {}
    for family, configs in FAMILIES.items():
        scores[family] = {}
        for config in configs:
            values = []
            for seed in SEEDS:
                identity = verify_run(config, seed)
                identities[f"{config}_seed{seed}"] = identity
                values.append(identity["best_validation_median"])
            scores[family][config] = values
    body = {"design_sha256": sha256(out / "design.json"), "criterion": "mean_of_seed_best_validation_medians",
            "scores": scores, "choices": choose_settings(scores), "run_identities": identities}
    path = out / "selection.json"
    if path.exists():
        previous = json.loads(path.read_text())
        previous_body = {k: v for k, v in previous.items() if k != "created_utc"}
        if previous_body != body:
            raise ValueError("Frozen validation selection or its inputs changed; refusing to replace it")
        return previous
    if any((out / "eval").glob("*")):
        raise ValueError("Selected test artifacts already exist without a frozen selection; refuse to create a replacement")
    body["created_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, body)
    print(f"Selection frozen BEFORE test evaluation: {body['choices']}", flush=True)
    return body


def selected_cache(out: Path, selection: dict, identity: dict, tracks: list[str]) -> pd.DataFrame | None:
    stem = f"{identity['config']}_seed{identity['seed']}_test"
    path, metadata_path = out / "eval" / f"{stem}.csv", out / "eval" / f"{stem}.json"
    if not path.exists() or not metadata_path.exists():
        return None
    frame, metadata = pd.read_csv(path), json.loads(metadata_path.read_text())
    validate_evaluation(frame, metadata, identity, tracks)
    if (metadata.get("checkpoint_sha256") != identity["hashes"]["best.pt"]
            or metadata.get("selection_sha256") != sha256(out / "selection.json")
            or metadata.get("csv_sha256") != sha256(path)):
        raise ValueError(f"{stem}: cached evaluation hashes differ from the frozen selection/checkpoint")
    return frame


def evaluate_selected(out: Path, device: str):
    # select() uses validation logs/checkpoints only; no new test inference precedes this write.
    selection = select(out)
    design = json.loads((out / "design.json").read_text())
    tracks = design["test_tracks"]
    for family, config in selection["choices"].items():
        for seed in SEEDS:
            identity = selection["run_identities"][f"{config}_seed{seed}"]
            if config == FAMILIES[family][0]:
                _, frozen = original_evaluation(identity, tracks)
                if frozen != design["original_evaluations"][f"{config}_seed{seed}"]:
                    raise ValueError("Original evaluation changed after design freeze")
                print(f"Reuse frozen original evaluation: {config} seed {seed}", flush=True)
                continue
            if selected_cache(out, selection, identity, tracks) is not None:
                print(f"Reuse verified selected evaluation: {config} seed {seed}", flush=True)
                continue
            evaluate_one(out, identity, tracks, device)
    summarize(out)


def evaluate_one(out: Path, identity: dict, tracks: list[str], device: str):
    import torch
    from dwmss.data import STEMS, StemStore
    from dwmss.metrics import si_sdr, usdr
    from dwmss.model import MaskNet
    from dwmss.separator import separate
    from dwmss.stft import DualWindowSTFT

    if not (out / "selection.json").is_file():
        raise ValueError("Write and verify the validation-only selection before test evaluation")
    selection = json.loads((out / "selection.json").read_text())
    if identity["config"] not in selection["choices"].values():
        raise ValueError("Test inference is restricted to selected configurations")
    expected = selection["run_identities"].get(f"{identity['config']}_seed{identity['seed']}")
    if expected != identity or sha256(Path(identity["run"]) / "best.pt") != identity["hashes"]["best.pt"]:
        raise ValueError("Selected checkpoint identity changed")
    checkpoint = torch.load(Path(identity["run"]) / "best.pt", map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    stft = DualWindowSTFT(cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"]).to(device)
    model = MaskNet(**cfg["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    store = StemStore(cfg["data"]["musdb_root"], "test", tracks)
    rows = []
    for i, track in enumerate(tracks):
        n = store.lengths[track]
        vocal = store.read(track, "vocals", 0, n)
        accompaniment = sum(store.read(track, stem, 0, n) for stem in STEMS[1:])
        mixture = (vocal + accompaniment).astype(np.float32)
        estimate = separate(model, stft, mixture)
        quality, mixture_quality = si_sdr(estimate, vocal), si_sdr(mixture, vocal)
        rows.append({"track": track, "seconds": n / 44100, "si_sdr": quality,
                     "usdr": usdr(estimate, vocal), "si_sdr_mixture": mixture_quality,
                     "si_sdri": quality - mixture_quality})
        print(f"{identity['config']} seed {identity['seed']} [{i + 1}/50] {track}: {quality:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    metadata = {"config": identity["config"], "seed": identity["seed"], "step": identity["best_step"],
                "ckpt": "best.pt", "subset": "test", "n_tracks": 50,
                "checkpoint_sha256": identity["hashes"]["best.pt"],
                "selection_sha256": sha256(out / "selection.json"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": device,
                "evaluation_utc": datetime.now(timezone.utc).isoformat()}
    validate_evaluation(frame, metadata, identity, tracks)
    path = out / "eval" / f"{identity['config']}_seed{identity['seed']}_test.csv"
    atomic_csv(path, frame)
    metadata["csv_sha256"] = sha256(path)
    atomic_json(path.with_suffix(".json"), metadata)


def paired(first: pd.Series, second: pd.Series) -> dict:
    if len(first) != 50 or len(second) != 50 or set(first.index) != set(second.index):
        raise ValueError("Paired inference requires identical complete 50-song sets")
    d = (first.sort_index() - second.sort_index()).to_numpy()
    if not np.isfinite(d).all():
        raise ValueError("Paired differences must be finite")
    samples = np.random.default_rng(0).choice(d, size=(10000, len(d)))
    med_lo, med_hi = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    mean_lo, mean_hi = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return {"songs": 50, "median": float(np.median(d)), "median_ci_lo": float(med_lo),
            "median_ci_hi": float(med_hi), "mean": float(d.mean()), "mean_ci_lo": float(mean_lo),
            "mean_ci_hi": float(mean_hi), "first_better": int((d > 0).sum())}


def summarize(out: Path):
    selection = select(out)
    design = json.loads((out / "design.json").read_text())
    frames, originals, rows, per_track = {}, {}, [], []
    for family, config in selection["choices"].items():
        selected, original = [], []
        for seed in SEEDS:
            identity = selection["run_identities"][f"{config}_seed{seed}"]
            original_identity = selection["run_identities"][f"{FAMILIES[family][0]}_seed{seed}"]
            old, _ = original_evaluation(original_identity, design["test_tracks"])
            frame = old if config == FAMILIES[family][0] else selected_cache(out, selection, identity, design["test_tracks"])
            if frame is None:
                raise ValueError(f"Selected test evaluation is pending: {config} seed {seed}")
            selected.append(frame.assign(seed=seed))
            original.append(old.assign(seed=seed))
        full = pd.concat(selected)
        frames[family] = full.groupby("track")["si_sdr"].mean().sort_index()
        originals[family] = pd.concat(original).groupby("track")["si_sdr"].mean().sort_index()
        cfg = design["resolved_configs"][config]
        rows.append({"family": family, "selected_config": config, "learning_rate": cfg["train"]["lr"],
                     "validation_selection_score": float(np.mean(selection["scores"][family][config])),
                     "si_sdr": float(frames[family].median()),
                     "seed0_si_sdr": float(selected[0].si_sdr.median()), "seed1_si_sdr": float(selected[1].si_sdr.median()),
                     "seed0_best_step": selection["run_identities"][f"{config}_seed0"]["best_step"],
                     "seed1_best_step": selection["run_identities"][f"{config}_seed1"]["best_step"]})
        per_track.extend({"family": family, "selected_config": config, "track": track, "si_sdr_seed_mean": float(value)}
                         for track, value in frames[family].items())
    comparisons = [dict(comparison="selected_dual_minus_selected_wide_short", **paired(frames["dual"], frames["wide_short"])),
                   dict(comparison="original_dual_minus_original_wide_short", **paired(originals["dual"], originals["wide_short"]))]
    comparisons.extend(dict(comparison=f"selected_minus_original_{family}", **paired(frames[family], originals[family]))
                       for family in FAMILIES)
    atomic_csv(out / "summary.csv", pd.DataFrame(rows))
    atomic_csv(out / "per_track.csv", pd.DataFrame(per_track))
    atomic_csv(out / "paired.csv", pd.DataFrame(comparisons))
    lines = ["# Bounded learning-rate check at 2.9 ms", "", "Status: complete.", "",
             "Exploratory follow-up motivated by already observed original test results; this reuses the same test set. "
             "The two learning-rate candidates and selection rule were frozen before new training. "
             "This is a symmetric one-candidate sensitivity check, not an optimization study.", "",
             "Each architecture received the existing 0.001 setting plus one 0.0003 setting, each with seeds 0/1 "
             "and 200000 updates. Weight decay remains 0.01 and all other settings are identical. "
             "Selection uses the mean of the two seeds' own best validation median SI-SDR. "
             "One setting is chosen per architecture; ties within 1e-9 dB favor the original. selection.json was written before new test inference.", "",
             "| architecture | selected setting | learning rate | selection validation score | test SI-SDR | test seed 0 / 1 | best steps 0 / 1 |",
             "|---|---|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['family']} | {row['selected_config']} | {row['learning_rate']:g} | "
                     f"{row['validation_selection_score']:.3f} | {row['si_sdr']:.2f} | "
                     f"{row['seed0_si_sdr']:.2f} / {row['seed1_si_sdr']:.2f} | "
                     f"{row['seed0_best_step']} / {row['seed1_best_step']} |")
    lines += ["", "Validation selection score averages seed maxima; test SI-SDR averages seeds within each song "
              "before the 50-song median. These aggregation orders are different.", "",
              "| comparison | median paired gain [95% CI] | mean paired gain [95% CI] | first better |",
              "|---|---|---|---:|"]
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['median']:+.2f} [{row['median_ci_lo']:+.2f}, {row['median_ci_hi']:+.2f}] | "
                     f"{row['mean']:+.2f} [{row['mean_ci_lo']:+.2f}, {row['mean_ci_hi']:+.2f}] | {row['first_better']} / 50 |")
    lines += ["", "Song bootstrap: 10000 draws, seed 0, 95% percentile intervals. "
              "Intervals do not include training-seed uncertainty. All 50 songs remain in the comparison.", "",
              "The selected checkpoint per seed is validation best.pt, so early stopping was already applied. "
              "Changing the learning rate can reveal recipe sensitivity; neither outcome proves the cause of the original "
              "validation deterioration, nor excludes better short-window architectures or training strategies. "
              "Only 2.9 ms was checked. Architecture, model MACs and algorithmic latency are unchanged, "
              "but no new CPU timing or physical end-to-end latency claim is made.", "",
              "All new evaluations live in this directory; original MAC-matched reports and evaluations remain intact. "
              "See design.json, selection.json and evaluation sidecars for source/checkpoint/CSV hashes."]
    atomic_text(out / "summary.md", "\n".join(lines) + "\n")
    atomic_json(out / "summary_provenance.json", {
        "design_sha256": sha256(out / "design.json"), "selection_sha256": sha256(out / "selection.json"),
        "selected_evaluations": {str(path.relative_to(out)): sha256(path)
                                 for path in sorted((out / "eval").glob("*")) if path.is_file()},
        "outputs": {name: sha256(out / name) for name in ("summary.csv", "per_track.csv", "paired.csv", "summary.md")}})
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    actions = ap.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--select", action="store_true")
    actions.add_argument("--evaluate-selected", action="store_true")
    actions.add_argument("--summarize", action="store_true")
    actions.add_argument("--verify-run", metavar="RUN")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.freeze:
        freeze(args.out)
    elif args.check:
        check_design(args.out)
        print("Frozen recipe-check design verified")
    elif args.select:
        select(args.out)
    elif args.evaluate_selected:
        evaluate_selected(args.out, args.device)
    elif args.summarize:
        summarize(args.out)
    else:
        check_design(args.out)
        allowed = {f"{config}_seed{seed}": (config, seed) for configs in FAMILIES.values() for config in configs for seed in SEEDS}
        if args.verify_run not in allowed:
            raise ValueError("Run must be one of the prescribed candidates/seeds")
        config, seed = allowed[args.verify_run]
        identity = verify_run(config, seed)
        atomic_json(args.out / "training_provenance" / f"{args.verify_run}.json", identity)
        print(f"Verified complete training: {args.verify_run}")


if __name__ == "__main__":
    main()
