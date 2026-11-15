"""Audit saved training trajectories without training or evaluating any model.

Usage: python -I scripts/analyze_training_diagnostics.py
Outputs are isolated in results/training_diagnostics/. All twelve prescribed runs
must be complete. Logged training losses are sampled minibatches, not held-out
training-set performance, and cannot establish an overfitting mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dwmss.data import VALIDATION_TRACKS  # noqa: E402
from dwmss.plotstyle import AQUA, BLUE, INK, INK2, MUTED, ORANGE, apply_style, plt  # noqa: E402

CONFIGS = ("B_sym256", "C_dual", "E_sym128", "D_dual_2p9ms", "B_sym256_compute", "E_sym128_compute")
SEEDS = (0, 1)
MAX_STEPS = 200_000
BIN_STEPS = 5_000
GROUPS = (("5.8 ms", ("B_sym256", "C_dual", "B_sym256_compute")),
          ("2.9 ms", ("E_sym128", "D_dual_2p9ms", "E_sym128_compute")))
STYLE = {"B_sym256": (ORANGE, "B short"), "C_dual": (AQUA, "C dual"),
         "E_sym128": (ORANGE, "E short"), "D_dual_2p9ms": (AQUA, "D dual"),
         "B_sym256_compute": (BLUE, "B wide short"), "E_sym128_compute": (BLUE, "E wide short")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonfinite(value, context: str):
    """JSON accepts NaN/Infinity by default; reject them even in optional fields."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context}: non-finite numeric value")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{context}[{index}]")


def read_json(path: Path):
    value = json.loads(path.read_text())
    reject_nonfinite(value, str(path))
    return value


def read_log(path: Path, log_every: int, val_every: int, max_steps: int = MAX_STEPS):
    train, validation = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{number}: empty log record")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: malformed JSON") from exc
        reject_nonfinite(record, f"{path}:{number}")
        if not isinstance(record, dict) or type(record.get("step")) is not int:
            raise ValueError(f"{path}:{number}: expected an integer step")
        if "loss" in record and "val_si_sdr_median" not in record:
            required = ("loss", "grad_norm", "lr")
            destination = train
        elif "val_si_sdr_median" in record and "loss" not in record:
            required = ("val_si_sdr_median", "val_si_sdr_mean", "val_si_sdri_median", "best", "improved")
            destination = validation
        else:
            raise ValueError(f"{path}:{number}: unknown or ambiguous record type")
        if any(key not in record for key in required):
            raise ValueError(f"{path}:{number}: missing fields {set(required) - set(record)}")
        if any(type(record[key]) not in (int, float) for key in required if key != "improved"):
            raise ValueError(f"{path}:{number}: metric fields must be numeric")
        if "improved" in required and type(record["improved"]) is not bool:
            raise ValueError(f"{path}:{number}: improved must be boolean")
        if destination and record["step"] <= destination[-1]["step"]:
            raise ValueError(f"{path}:{number}: duplicate or unordered {('training' if destination is train else 'validation')} step")
        destination.append(record)
    for records, interval, name in ((train, log_every, "training"), (validation, val_every, "validation")):
        expected = list(range(interval, max_steps + 1, interval))
        if max_steps % interval:
            expected.append(max_steps)
        # Training writes only on its logging cadence; a non-cadence final step
        # is not expected there. Production 200k is divisible by both intervals.
        if name == "training" and max_steps % interval:
            expected.pop()
        if [record["step"] for record in records] != expected:
            raise ValueError(f"{path}: missing or unexpected {name} steps; expected complete cadence through {max_steps}")
    if not train or not validation:
        raise ValueError(f"{path}: training and validation records are required")
    if any(record["loss"] < 0 or record["grad_norm"] < 0 or record["lr"] <= 0 for record in train):
        raise ValueError(f"{path}: invalid loss, gradient norm, or learning rate")
    running_best = -math.inf
    for record in validation:
        improved = record["val_si_sdr_median"] > running_best
        running_best = max(running_best, record["val_si_sdr_median"])
        if record["improved"] != improved or not math.isclose(record["best"], running_best, abs_tol=1e-8):
            raise ValueError(f"{path}: inconsistent running-best metadata at step {record['step']}")
    return pd.DataFrame(train), pd.DataFrame(validation)


def window_summary(frame: pd.DataFrame, grad_clip: float) -> dict:
    if frame.empty:
        raise ValueError("A diagnostic training window is empty")
    return {"sampled_batches": len(frame), "loss_median": float(frame.loss.median()),
            "loss_q25": float(frame.loss.quantile(.25)), "loss_q75": float(frame.loss.quantile(.75)),
            "grad_norm_median": float(frame.grad_norm.median()),
            "grad_norm_p95": float(frame.grad_norm.quantile(.95)),
            "clipped_logged_batches": int((frame.grad_norm > grad_clip).sum()),
            "clipping_fraction_logged": float((frame.grad_norm > grad_clip).mean())}


def check_checkpoint(checkpoint: dict, metadata: dict, validation: pd.DataFrame, evaluation: dict,
                     config: str, seed: int):
    # idxmax picks the earliest maximum, matching train.py's strict improvement.
    best = validation.loc[validation.val_si_sdr_median.idxmax()]
    if checkpoint.get("config") != metadata["config"] or checkpoint.get("seed") != seed:
        raise ValueError(f"{config} seed {seed}: checkpoint identity/config differs from metadata")
    if checkpoint.get("step") != int(best.step) or not math.isclose(
            float(checkpoint.get("best", math.nan)), float(best.val_si_sdr_median), abs_tol=1e-7):
        raise ValueError(f"{config} seed {seed}: best.pt does not match the maximum validation log")
    if (evaluation.get("config") != config or evaluation.get("ckpt") != "best.pt"
            or evaluation.get("step") != checkpoint["step"] or evaluation.get("n_tracks") != 50
            or evaluation.get("subset") != "test"):
        raise ValueError(f"{config} seed {seed}: evaluation does not identify this 50-song best checkpoint")
    return best


def analyze_run(run: Path, eval_dir: Path, config: str, seed: int, max_steps: int = MAX_STEPS):
    paths = {"log": run / "log.jsonl", "meta": run / "meta.json", "done": run / "DONE",
             "best_checkpoint": run / "best.pt", "evaluation_metadata": eval_dir / f"{run.name}_test.json"}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Required training diagnostic source is missing: {path}")
    metadata = read_json(paths["meta"])
    if metadata.get("seed") != seed or metadata.get("config", {}).get("name") != config:
        raise ValueError(f"{run}: run identity differs from metadata")
    tc = metadata["config"]["train"]
    if tc.get("max_steps") != max_steps or paths["done"].read_text().strip() != f"finished {max_steps} steps":
        raise ValueError(f"{run}: expected a completed {max_steps}-update run")
    if len(metadata.get("valid_tracks", [])) != 14 or set(metadata["valid_tracks"]) != set(VALIDATION_TRACKS):
        raise ValueError(f"{run}: expected the official complete 14-song validation split")
    for key in ("log_every", "val_every"):
        if type(tc.get(key)) is not int or tc[key] <= 0:
            raise ValueError(f"{run}: invalid {key}")
    grad_clip = float(tc["grad_clip"])
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError(f"{run}: invalid gradient clipping threshold")
    train, validation = read_log(paths["log"], tc["log_every"], tc["val_every"], max_steps)
    evaluation = read_json(paths["evaluation_metadata"])
    checkpoint = torch.load(paths["best_checkpoint"], map_location="cpu", weights_only=False)
    best = check_checkpoint(checkpoint, metadata, validation, evaluation, config, seed)
    del checkpoint
    final = validation.iloc[-1]
    row = {"config": config, "seed": seed, "params": metadata["params"], "completed_steps": max_steps,
           "validation_points": len(validation), "best_step": int(best.step),
           "best_val_si_sdr": float(best.val_si_sdr_median), "final_val_si_sdr": float(final.val_si_sdr_median),
           "best_to_final_val_drop_db": float(best.val_si_sdr_median - final.val_si_sdr_median),
           "validation_points_after_best": int((validation.step > best.step).sum()),
           "evaluation_step": evaluation["step"], "best_checkpoint_and_evaluation_verified": True,
           "grad_clip_threshold": grad_clip, "initial_lr": float(tc["lr"]),
           "weight_decay": float(tc["weight_decay"]), "log_every": tc["log_every"],
           "val_every": tc["val_every"], "early_window_end_step": max_steps // 5,
           "late_window_start_exclusive": max_steps * 4 // 5,
           "sampled_batches": len(train), "clipped_logged_batches": int((train.grad_norm > grad_clip).sum()),
           "clipping_fraction_logged": float((train.grad_norm > grad_clip).mean())}
    for prefix, frame in (("early", train[train.step <= max_steps // 5]),
                          ("late", train[train.step > max_steps * 4 // 5])):
        row.update({f"{prefix}_{key}": value for key, value in window_summary(frame, grad_clip).items()})
    train = train.assign(config=config, seed=seed, grad_clip_threshold=grad_clip,
                         clipped_logged=train.grad_norm > grad_clip)
    validation = validation.assign(config=config, seed=seed,
                                   selected_best=validation.step == int(best.step),
                                   final_checkpoint=validation.step == max_steps)
    sources = [{"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
               for role, path in paths.items()]
    return row, train, validation, metadata, sources


def binned_training(train: pd.DataFrame) -> pd.DataFrame:
    frame = train.copy()
    frame["bin_end_step"] = ((frame.step - 1) // BIN_STEPS + 1) * BIN_STEPS
    return frame.groupby(["config", "seed", "bin_end_step"], as_index=False).agg(
        sampled_batches=("loss", "size"), loss_median=("loss", "median"),
        grad_norm_median=("grad_norm", "median"), clipping_fraction_logged=("clipped_logged", "mean"),
        lr_median=("lr", "median"))


def make_plots(summary: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, out: Path):
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), sharex=True, sharey="row")
    for row_index, (latency, configs) in enumerate(GROUPS):
        for column, seed in enumerate(SEEDS):
            ax = axes[row_index, column]
            for config in configs:
                color, label = STYLE[config]
                v = validation[(validation.config == config) & (validation.seed == seed)]
                selected = summary[(summary.config == config) & (summary.seed == seed)].iloc[0]
                ax.plot(v.step / 1000, v.val_si_sdr_median, color=color, lw=1.8, label=label)
                ax.scatter([selected.best_step / 1000], [selected.best_val_si_sdr], marker="*", s=120,
                           color=color, edgecolors="white", linewidths=.6, zorder=4)
                ax.scatter([MAX_STEPS / 1000], [selected.final_val_si_sdr], marker="o", s=35,
                           facecolors="none", edgecolors=color, linewidths=1.4, zorder=4)
            ax.set_title(f"{latency} · seed {seed}")
            ax.set_ylabel("Validation SI-SDR (dB, song median)")
            ax.grid(True, alpha=.6)
            ax.legend(loc="lower right")
            ax.set_xlim(0, 205)
    for ax in axes[-1]:
        ax.set_xlabel("Training updates (thousands)")
    fig.suptitle("Validation trajectories and the checkpoints actually evaluated", x=.08, y=.99,
                 ha="left", color=INK, fontsize=15, fontweight="bold")
    fig.text(.08, .94, "14 validation songs · star = selected best.pt · open circle = final 200k validation\n"
             "Each seed is shown separately; current test results already use the validation-selected checkpoint.",
             color=INK2, fontsize=10.5, va="top")
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.08, hspace=.28, wspace=.16)
    fig.savefig(out / "fig_validation_trajectories.png")
    plt.close(fig)

    bins = binned_training(train)
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 10.2), sharex=True)
    for column, (latency, configs) in enumerate(GROUPS):
        for config in configs:
            color, label = STYLE[config]
            for seed in SEEDS:
                values = bins[(bins.config == config) & (bins.seed == seed)]
                linestyle = "-" if seed == 0 else (0, (4, 2))
                for row_index, field in enumerate(("loss_median", "grad_norm_median", "lr_median")):
                    axes[row_index, column].plot(values.bin_end_step / 1000, values[field], color=color,
                                                ls=linestyle, lw=1.7, label=label if seed == 0 else None)
        axes[0, column].set_title(latency)
        # legend above the panel, so that it cannot cover the curves
        axes[0, column].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, borderaxespad=0.2)
        threshold = train[train.config.isin(configs)].grad_clip_threshold.unique()
        for clip in threshold:
            axes[1, column].axhline(clip, color=MUTED, lw=1, ls=":")
            # label the threshold on the line itself; the curve colors are those of the legend above
            axes[1, column].annotate(f"clip threshold {clip:g}", (200, clip), xytext=(0, -5), textcoords="offset points",
                                     ha="right", va="top", fontsize=9, color=INK2)
        axes[1, column].set_yscale("log")
        axes[2, column].set_yscale("log")
        axes[2, column].set_xlabel("Training updates (thousands)")
        for row_index, ylabel in enumerate(("Sampled minibatch loss (median)", "Pre-clipping gradient norm (median)",
                                           "Learning rate (median)")):
            axes[row_index, column].set_ylabel(ylabel)
            axes[row_index, column].grid(True, alpha=.6)
            axes[row_index, column].set_xlim(0, 205)
    fig.suptitle("Optimization diagnostics from sampled training batches", x=.08, y=.99,
                 ha="left", color=INK, fontsize=15, fontweight="bold")
    fig.text(.08, .948, "Medians within 5k-update bins · solid = seed 0, dashed = seed 1 (learning-rate curves overlap)\n"
             "Losses come from changing augmented minibatches; these are not held-out train losses or generalization gaps.",
             color=INK2, fontsize=10.5, va="top")
    fig.subplots_adjust(left=.08, right=.98, top=.855, bottom=.065, hspace=.22, wspace=.19)
    fig.savefig(out / "fig_optimization_diagnostics.png")
    plt.close(fig)
    return bins


def write_report(summary: pd.DataFrame, out: Path):
    rows = summary.set_index(["config", "seed"])
    lines = ["# Training diagnostics for the short-window compute controls", "",
             "All twelve prescribed runs completed 200,000 updates. Every best.pt checkpoint's config, seed, "
             "step and validation score match the maximum in its saved validation log (first maximum on ties). "
             "Every existing 50-song test-evaluation sidecar identifies that best.pt and the same step. "
             "The 14 validation songs are identical across runs and match the official validation split "
             "recorded in dwmss/data.py. No training or model evaluation was rerun.", "",
             "**The current test comparisons already use validation-selected best.pt checkpoints.** Later "
             "validation deterioration therefore does not mean the final 200k checkpoint was mistakenly tested. "
             "For older runs the evaluation sidecar establishes checkpoint filename/step agreement, not a "
             "retrospective cryptographic proof of the checkpoint bytes used at evaluation time. The source "
             "hashes in this report record their state when this diagnostic was produced.", "",
             "## Validation trajectories", "",
             "| config | seed | best update | best validation SI-SDR | final validation SI-SDR | best − final | logged batches clipped |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.config} | {row.seed} | {row.best_step:,} | {row.best_val_si_sdr:.3f} | "
                     f"{row.final_val_si_sdr:.3f} | {row.best_to_final_val_drop_db:.3f} | "
                     f"{row.clipping_fraction_logged:.2%} |")
    lines += ["", "![Validation trajectories](fig_validation_trajectories.png)", "",
              "## What the wider controls show", ""]
    for name, baseline in (("B_sym256_compute", "B_sym256"), ("E_sym128_compute", "E_sym128")):
        controls = [rows.loc[(name, seed)] for seed in SEEDS]
        bases = [rows.loc[(baseline, seed)] for seed in SEEDS]
        best_steps = " / ".join(f"{int(row.best_step):,}" for row in controls)
        drops = " / ".join(f"{row.best_to_final_val_drop_db:.3f}" for row in controls)
        gaps = " / ".join(f"{(control.best_val_si_sdr - base.best_val_si_sdr):+.3f}"
                          for control, base in zip(controls, bases))
        lines.append(f"- **{name}**: best validation at {best_steps} updates (seeds 0 / 1); "
                     f"best-to-final decline {drops} dB. Its best validation score is {gaps} dB "
                     f"relative to {baseline}'s separately selected best for the same seed.")
    lines += ["", "The trajectories show validation deterioration and sensitivity to the shared training recipe. "
              "They do not identify its cause. Overfitting, optimization difficulty, augmentation variability and "
              "architecture limitations are competing explanations; these sampled losses cannot distinguish them. "
              "In particular, a later validation decline alone cannot explain away the existing test result: "
              "the earlier best checkpoint was already selected.", "",
              "## Descriptive training-batch summaries", "",
              "Early = updates 1–40,000; late = updates 160,001–200,000. These fixed windows each contain "
              "only the minibatches recorded at the configured logging cadence, with changing random crops, "
              "gains and remixes. Values summarize logged batches; they are not held-out training-set loss "
              "or evidence of a train/validation generalization gap. Gradient norms are returned by "
              "clip_grad_norm_ before clipping; a logged batch is counted as clipped only if its norm is "
              "strictly above the configured threshold. The clipping fraction is not an estimate of the "
              "fraction over all optimizer updates unless the logged batches are representative.", "",
              "| config | seed | early / late loss median | early / late grad-norm median | early / late logged clipping |",
              "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.config} | {row.seed} | {row.early_loss_median:.4f} / {row.late_loss_median:.4f} | "
                     f"{row.early_grad_norm_median:.3f} / {row.late_grad_norm_median:.3f} | "
                     f"{row.early_clipping_fraction_logged:.2%} / {row.late_clipping_fraction_logged:.2%} |")
    lines += ["", "![Optimization diagnostics](fig_optimization_diagnostics.png)", "",
              "## Is a bounded follow-up warranted?", "",
              "Yes. The resulting bounded protocol is recorded in "
              "[`EXPERIMENTS.md` section 4](../../EXPERIMENTS.md#bounded-learning-rate-check-at-29-ms) "
              "and its [frozen design](../recipe_check/design.md): only the 2.9 ms D dual and E wide-short "
              "architectures are compared, with the existing 1e-3 learning rate and one added 3e-4 setting. "
              "Weight decay stays 0.01 and all other settings stay fixed. The existing two original-rate seeds "
              "per architecture are reused; exactly four fresh 200k-update runs cover the lower rate, with "
              "equal candidate/update budgets for D and E. Select one recipe per architecture by the mean "
              "of its two seeds' best validation medians, with ties favoring the original rate, before any "
              "new test evaluation. There is no broader learning-rate/weight-decay or 5.8 ms search. "
              "The test set has already informed this research direction, so its eventual test result remains "
              "an exploratory follow-up, not an untouched confirmatory test. This diagnostic script launches "
              "no training. The original fixed-recipe comparison remains unchanged.", "",
              "## Files and reproducibility", "",
              "- `per_run.csv`: best/final validation, checkpoint checks, clipping fractions and fixed-window summaries.",
              "- `validation_curves.csv`: every saved validation point, with selected-best and final-point flags.",
              "- `sampled_training_batches.csv`: every logged batch with loss, pre-clipping norm, rate and clip flag.",
              "- `training_bins.csv`: descriptive medians in fixed 5k-update bins used by the optimization figure.",
              "- `provenance.json`: SHA-256 of each input log, metadata, DONE marker, best checkpoint, evaluation "
              "sidecar, and this script; hashes of generated tables, report and figures.", "",
              "Reproduce: `python -I scripts/analyze_training_diagnostics.py`.", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=REPO / "runs")
    parser.add_argument("--eval-dir", type=Path, default=REPO / "results" / "eval")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "training_diagnostics")
    args = parser.parse_args()
    summaries, training, validation, sources = [], [], [], []
    validation_tracks = None
    for config in CONFIGS:
        for seed in SEEDS:
            run = args.runs_root / f"{config}_seed{seed}"
            row, train, valid, metadata, inputs = analyze_run(run, args.eval_dir, config, seed)
            tracks = tuple(sorted(metadata["valid_tracks"]))
            if validation_tracks is not None and tracks != validation_tracks:
                raise ValueError(f"{run}: different validation split")
            validation_tracks = tracks
            summaries.append(row)
            training.append(train)
            validation.append(valid)
            sources.extend({"run": run.name, **item} for item in inputs)
            print(f"verified {run.name}: best {row['best_step']:,}, "
                  f"best-to-final decline {row['best_to_final_val_drop_db']:.3f} dB", flush=True)
    summary = pd.DataFrame(summaries)
    train = pd.concat(training, ignore_index=True)
    valid = pd.concat(validation, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "per_run.csv", index=False)
    train.to_csv(args.out / "sampled_training_batches.csv", index=False)
    valid.to_csv(args.out / "validation_curves.csv", index=False)
    bins = make_plots(summary, train, valid, args.out)
    bins.to_csv(args.out / "training_bins.csv", index=False)
    write_report(summary, args.out)
    artifact_names = ("per_run.csv", "sampled_training_batches.csv", "validation_curves.csv", "training_bins.csv",
                      "summary.md", "fig_validation_trajectories.png", "fig_optimization_diagnostics.png")
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).resolve()),
                  "script_sha256": sha256(Path(__file__)),
                  "source_dependencies": {str(path.relative_to(REPO)): sha256(path)
                                          for path in (REPO / "dwmss" / "data.py", REPO / "dwmss" / "plotstyle.py")},
                  "validation_tracks": list(validation_tracks),
                  "configs": list(CONFIGS), "seeds": list(SEEDS), "completed_steps": MAX_STEPS,
                  "descriptive_training_windows": {"early_end_inclusive": 40_000, "late_start_exclusive": 160_000},
                  "inputs": sources,
                  "outputs": {name: sha256(args.out / name) for name in artifact_names}}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Saved training diagnostics to {args.out}", flush=True)


if __name__ == "__main__":
    main()
