"""Report the prespecified MAC-matched controls without mixing them into the main results.

Usage: python -I scripts/summarize_compute_matched.py [--require-complete]
The strict mode requires two complete 50-song evaluations per model and one same-job,
trained-weight, single-thread ONNX profile for all six models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dwmss.config import load_config  # noqa: E402
from scripts.select_compute_matched import SEEDS, analytic_description, make_design  # noqa: E402

CONFIGS = ("B_sym256", "B_sym256_compute", "C_dual", "E_sym128", "E_sym128_compute", "D_dual_2p9ms")
CONTROLS = ("B_sym256_compute", "E_sym128_compute")
COMPARISONS = (
    ("primary", "C_dual", "B_sym256_compute", "dual minus MAC-matched short, 5.8 ms"),
    ("primary", "D_dual_2p9ms", "E_sym128_compute", "dual minus MAC-matched short, 2.9 ms"),
    ("secondary", "B_sym256_compute", "B_sym256", "wider minus original short, 5.8 ms"),
    ("secondary", "E_sym128_compute", "E_sym128", "wider minus original short, 2.9 ms"),
    ("reference", "C_dual", "B_sym256", "original dual minus original short, 5.8 ms"),
    ("reference", "D_dual_2p9ms", "E_sym128", "original dual minus original short, 2.9 ms"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_evaluation(run: Path, eval_dir: Path, out: Path):
    """Call immediately after evaluation to associate its files with the checkpoint used."""
    run = run.resolve()
    valid = {f"{config}_seed{seed}": (config, seed) for config in CONTROLS for seed in SEEDS}
    if run.name not in valid or run.parent != REPO / "runs":
        raise ValueError("Record only a prescribed control run in this repository's runs directory")
    config, seed = valid[run.name]
    stem = f"{run.name}_test"
    checkpoint, csv_path, meta_path = run / "best.pt", eval_dir / f"{stem}.csv", eval_dir / f"{stem}.json"
    metadata = json.loads(meta_path.read_text())
    if metadata.get("config") != config or metadata.get("n_tracks") != 50 or metadata.get("ckpt") != "best.pt":
        raise ValueError("Stamp only a complete 50-song best.pt evaluation of this control")
    if not (run / "DONE").exists() or (run / "DONE").read_text().strip() != "finished 200000 steps":
        raise ValueError("Stamp only after the 200000-update training run has completed")
    provenance = {"config": config, "seed": seed, "checkpoint_path": str(checkpoint),
                  "evaluation_step": metadata.get("step"), "checkpoint_sha256": sha256(checkpoint),
                  "csv_sha256": sha256(csv_path), "json_sha256": sha256(meta_path)}
    path = out / "evaluation_provenance" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"recorded checkpoint/evaluation hashes: {path}")


def verify_provenance(provenance_path: Path, checkpoint: Path, csv_path: Path, meta_path: Path, metadata: dict):
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("checkpoint_path") != str(checkpoint.resolve()) or provenance.get("evaluation_step") != metadata.get("step"):
        raise ValueError(f"{provenance_path}: checkpoint/evaluation metadata differs")
    for key, path in (("checkpoint_sha256", checkpoint), ("csv_sha256", csv_path), ("json_sha256", meta_path)):
        if provenance.get(key) != sha256(path):
            raise ValueError(f"{provenance_path}: {path.name} changed after evaluation was recorded")


def load_evaluations(eval_dir: Path, provenance_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    complete, missing = {}, []
    for config in CONFIGS:
        frames = []
        for seed in SEEDS:
            stem = f"{config}_seed{seed}_test"
            path, meta_path = eval_dir / f"{stem}.csv", eval_dir / f"{stem}.json"
            if not path.exists() or not meta_path.exists():
                missing.append(stem)
                continue
            frame = pd.read_csv(path)
            meta = json.loads(meta_path.read_text())
            required = {"track", "si_sdr", "si_sdri"}
            if not required.issubset(frame.columns):
                raise ValueError(f"{path}: missing columns {required - set(frame.columns)}")
            if len(frame) != 50 or frame["track"].nunique() != 50:
                raise ValueError(f"{path}: expected exactly 50 distinct full test songs")
            if not np.isfinite(frame[["si_sdr", "si_sdri"]].to_numpy()).all():
                raise ValueError(f"{path}: non-finite quality scores")
            if meta.get("config") != config or meta.get("n_tracks") != 50 or meta.get("ckpt") != "best.pt":
                raise ValueError(f"{meta_path}: expected this config's 50-song best.pt evaluation")
            if config in CONTROLS and provenance_dir is not None:
                provenance = provenance_dir / f"{stem}.json"
                if not provenance.exists():
                    missing.append(f"{stem} (checkpoint provenance)")
                    continue
                checkpoint = REPO / "runs" / f"{config}_seed{seed}" / "best.pt"
                verify_provenance(provenance, checkpoint, path, meta_path, meta)
            frame["seed"] = seed
            frames.append(frame)
        if len(frames) == len(SEEDS):
            if set(frames[0]["track"]) != set(frames[1]["track"]):
                raise ValueError(f"{config}: the seed evaluations use different songs")
            complete[config] = pd.concat(frames, ignore_index=True)
    if complete:
        tracks = set(next(iter(complete.values()))["track"])
        if any(set(frame["track"]) != tracks for frame in complete.values()):
            raise ValueError("The models were not evaluated on the same 50 songs")
    return complete, missing


def load_profile(path: Path) -> tuple[pd.DataFrame, dict]:
    """Accept only the isolated, same-job profile; never substitute the main benchmark."""
    if not path.exists():
        return pd.DataFrame(), {}
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise ValueError(f"CPU profile needs its environment sidecar: {meta_path}")
    table = pd.read_csv(path)
    env = json.loads(meta_path.read_text())["env"]
    if env.get("threads") != 1:
        raise ValueError("The CPU profile must use one thread")
    table = table[table["backend"] == "onnx"].copy()
    if len(table) != len(CONFIGS) or set(table["config"]) != set(CONFIGS):
        raise ValueError("The CPU profile must contain each of the six models exactly once with ONNX Runtime")
    for _, row in table.iterrows():
        config = row["config"]
        if not str(row["weights"]).replace("\\", "/").endswith(f"{config}_seed0/best.pt"):
            raise ValueError(f"{config}: CPU timing must use the trained seed-0 best checkpoint")
        expected = analytic_description(load_config(str(REPO / "configs" / f"{config}.yaml")))
        if (int(row["params"]) != expected["params"]
                or abs(row["macs_per_frame_M"] - expected["macs_per_frame_M"]) > 1e-8
                or int(row["hop_samples"]) != expected["hop_samples"]):
            raise ValueError(f"{config}: profile architecture differs from the frozen design")
    return table.set_index("config"), env


def paired(first: pd.Series, second: pd.Series) -> dict:
    d = (first - second).dropna()
    if len(d) != 50:
        raise ValueError("A complete paired comparison requires the same 50 songs")
    rng = np.random.default_rng(0)
    samples = rng.choice(d.to_numpy(), size=(10000, len(d)))
    med_lo, med_hi = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    mean_lo, mean_hi = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return {"songs": len(d), "median": float(d.median()), "median_ci_lo": float(med_lo),
            "median_ci_hi": float(med_hi), "mean": float(d.mean()), "mean_ci_lo": float(mean_lo),
            "mean_ci_hi": float(mean_hi), "songs_first_better": int((d > 0).sum())}


def summarize(eval_dir: Path, out: Path, benchmark: Path, require_complete: bool = False):
    frozen = out / "design.json"
    if not frozen.exists() or json.loads(frozen.read_text()) != make_design():
        raise ValueError("Freeze and verify the MAC-matched design before summarizing")
    frames, missing = load_evaluations(eval_dir, out / "evaluation_provenance")
    profile, env = load_profile(benchmark)
    if require_complete and (missing or profile.empty):
        raise ValueError(f"Experiment incomplete; missing evaluations: {missing}; CPU profile present: {not profile.empty}")
    per_track, per_seed, rows, long_rows = {}, {}, [], []
    for config in CONFIGS:
        cfg = load_config(str(REPO / "configs" / f"{config}.yaml"))
        row = {"config": config, **analytic_description(cfg), "seeds": 0,
               "si_sdr_seed0": np.nan, "si_sdr_seed1": np.nan,
               "si_sdr_median": np.nan, "si_sdri_median": np.nan,
               "rtf": np.nan, "p99_us": np.nan, "deadline_miss_rate": np.nan}
        if config in frames:
            frame = frames[config]
            avg = frame.groupby("track")[["si_sdr", "si_sdri"]].mean()
            per_track[config] = avg["si_sdr"]
            per_seed[config] = {seed: frame[frame.seed == seed].set_index("track")["si_sdr"] for seed in SEEDS}
            row.update({"seeds": len(SEEDS), "si_sdr_median": float(avg["si_sdr"].median()),
                        "si_sdri_median": float(avg["si_sdri"].median())})
            for seed in SEEDS:
                row[f"si_sdr_seed{seed}"] = float(per_seed[config][seed].median())
            long_rows.extend({"config": config, "track": track, "si_sdr_seed_mean": float(value)}
                             for track, value in per_track[config].items())
        if not profile.empty:
            row.update({key: float(profile.loc[config, key]) for key in ("rtf", "p99_us", "deadline_miss_rate")})
        rows.append(row)
    comparisons = []
    for role, first, second, label in COMPARISONS:
        if first not in per_track or second not in per_track:
            continue
        seed_gains = [float((per_seed[first][a] - per_seed[second][b]).median()) for a in SEEDS for b in SEEDS]
        comparisons.append({"role": role, "first": first, "second": second, "label": label,
                            **paired(per_track[first], per_track[second]),
                            "seed_pair_median_min": min(seed_gains), "seed_pair_median_max": max(seed_gains)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(long_rows, columns=["config", "track", "si_sdr_seed_mean"]).to_csv(out / "per_track.csv", index=False)
    pd.DataFrame(comparisons, columns=["role", "first", "second", "label", "songs", "median", "median_ci_lo",
                                      "median_ci_hi", "mean", "mean_ci_lo", "mean_ci_hi", "songs_first_better",
                                      "seed_pair_median_min", "seed_pair_median_max"]).to_csv(out / "paired.csv", index=False)
    complete = not missing and not profile.empty
    lines = ["# MAC-matched short-window controls", "",
             f"Status: {'complete' if complete else 'partial; primary conclusions pending missing outputs'}.", "",
             "Design fixed before training: [`design.md`](design.md), full grid and resolved configurations in "
             "`design.json`. Only the short-window GRU width and its projections change. This compares how a fixed "
             "model-MAC budget is allocated; it does not hold parameter count or CPU runtime constant.", ""]
    if missing:
        lines += ["Missing evaluations (a config is reported only after both seeds have 50 songs): " + ", ".join(missing), ""]
    lines += ["## Quality and measured cost", "",
              "SI-SDR: median over all 50 full-length mono test songs after averaging seeds 0 and 1 per song. "
              "Per-seed entries are separate song medians, whose mean need not equal the seed-averaged song median.", "",
              "| config | hidden | params | MMAC/frame | GMAC/s | seeds | SI-SDR seed0 / seed1 | SI-SDR | ONNX RTF | P99 ms (hop ms) | late hops |",
              "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|"]
    for row in rows:
        q = f"{row['si_sdr_median']:.2f}" if row["seeds"] else "pending"
        seed_q = " / ".join(f"{row[f'si_sdr_seed{seed}']:.2f}" for seed in SEEDS) if row["seeds"] else "pending"
        rtf = f"{row['rtf']:.3f}" if np.isfinite(row["rtf"]) else "pending"
        timing = f"{row['p99_us'] / 1000:.3f} ({row['hop_samples'] / 44.1:.3f})" if np.isfinite(row["p99_us"]) else "pending"
        late = f"{row['deadline_miss_rate']:.2%}" if np.isfinite(row["deadline_miss_rate"]) else "pending"
        lines.append(f"| {row['config']} | {row['hidden']} | {row['params']:,} | {row['macs_per_frame_M']:.3f} | "
                     f"{row['gmacs_per_s']:.3f} | {row['seeds']} | {seed_q} | {q} | {rtf} | {timing} | {late} |")
    lines += ["", "## Paired comparisons", "",
              "Positive differences favor the first model. The primary comparisons test whether the dual window "
              "retains a gain after allocating comparable model MACs to a larger short-window network.", "",
              "| role | first minus second | songs | median difference [95% CI] | mean difference [95% CI] | first better | seed-pair median range |",
              "|---|---|---:|---|---|---:|---|"]
    for row in comparisons:
        lines.append(f"| {row['role']} | {row['first']} − {row['second']} | {row['songs']} | "
                     f"{row['median']:+.2f} [{row['median_ci_lo']:+.2f}, {row['median_ci_hi']:+.2f}] | "
                     f"{row['mean']:+.2f} [{row['mean_ci_lo']:+.2f}, {row['mean_ci_hi']:+.2f}] | "
                     f"{row['songs_first_better']} / {row['songs']} | "
                     f"{row['seed_pair_median_min']:+.2f} … {row['seed_pair_median_max']:+.2f} |")
    lines += ["", "Intervals resample songs 10,000 times with seed 0. They exclude training-seed variation; "
              "the last column gives all four seed-pair medians separately. Per-song seed-averaged values are in "
              "`per_track.csv`; all comparison statistics are in `paired.csv`.", "",
              "## CPU measurement and scope", ""]
    if env:
        lines += ["One job, one CPU thread, ONNX Runtime and trained seed-0 checkpoints for all six models. "
                  "The full per-hop STFT–network–iSTFT pipeline is timed. Environment: "
                  + ", ".join(f"{key}={env[key]}" for key in ("cpu", "host", "slurm_job_id", "threads", "onnxruntime", "seconds_timed") if key in env) + ".", ""]
    else:
        lines += ["The isolated same-job CPU profile has not been produced yet. No timings from other jobs are substituted.", ""]
    lines += ["MAC matching is prespecified; CPU time is reported afterwards and does not select the controls. "
              "The wider controls have more parameters and can require different training wall-clock time despite "
              "the same 200,000-update budget. Results test this resource allocation under the shared training recipe; "
              "they do not establish that either architecture is optimal at a given CPU budget. "
              "Algorithmic latency is determined by the synthesis window. Audio-device end-to-end latency is not measured. "
              "Each new control's evaluation is associated with SHA-256 hashes of its checkpoint and result files; "
              "the report rejects artifacts changed after recording."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, default=REPO / "results" / "eval")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "compute_matched")
    ap.add_argument("--benchmark", type=Path, help="isolated same-job profile CSV; defaults to OUT/latency_benchmark.csv")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--record-evaluation", type=Path, metavar="RUN", help="record checkpoint and result hashes immediately after this control's evaluation")
    args = ap.parse_args()
    if args.record_evaluation is not None:
        if json.loads((args.out / "design.json").read_text()) != make_design():
            raise ValueError("The frozen design must match before recording an evaluation")
        record_evaluation(args.record_evaluation, args.eval_dir, args.out)
        return
    summarize(args.eval_dir, args.out, args.benchmark or args.out / "latency_benchmark.csv", args.require_complete)


if __name__ == "__main__":
    main()
