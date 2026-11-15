"""Redraw the failure-analysis figures from the saved tables, without re-running inference.

`scripts/analyze_failures.py` hashes itself into its cache identity, so its plotting code cannot be edited
without recomputing every model output. This script reads only what that script saved (paired.csv, cases.csv,
blocks/*.json, per_track.csv) and redraws the three figures with legends placed outside the panels. The numbers
are untouched. The figure hashes in summary_provenance.json are updated and the change is recorded there.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dwmss.plotstyle import AQUA, BLUE, GRAY, INK, INK2, ORANGE, apply_style, plt  # noqa: E402

OUT = REPO / "results" / "failure_analysis"
PAIRS = [("C_dual", "B_sym256", "C − B at 5.8 ms"), ("D_dual_2p9ms", "E_sym128", "D − E at 2.9 ms")]
FIGURES = ("fig_activity_sensitivity.png", "fig_worst_C_B.png", "fig_worst_D_E.png")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minus(value: float) -> str:
    return f"{int(value)}".replace("-", "−")


def activity_figure(paired: pd.DataFrame, songs: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.8), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.10, hspace=0.62)
    for ax, (_, _, label) in zip(axes[:, 0], PAIRS):
        rows = paired[(paired.pair == label) & paired.metric.isin(["whole_si_sdr", "active_si_sdr"])]
        # full-song result first, then the activity thresholds in the saved order
        rows = pd.concat([rows[rows.metric == "whole_si_sdr"], rows[rows.metric == "active_si_sdr"]])
        labels = ["Full song" if r.metric == "whole_si_sdr"
                  else f"Active: peak−{int(r.relative_db)} dB\nfloor {minus(r.floor_dbfs)} dBFS" for r in rows.itertuples()]
        x = np.arange(len(rows))
        ax.errorbar(x, rows["median"], yerr=[rows["median"] - rows.median_lo, rows.median_hi - rows["median"]],
                    fmt="o", color=AQUA, capsize=4, markersize=6)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Paired SI-SDR gain (dB)")
        ax.set_title(label)
        ax.grid(axis="y")
    fig.text(0.03, 0.965, "Full-song and reference-active diagnostics", fontsize=15, fontweight="bold",
             color=INK, va="top")
    fig.text(0.03, 0.915, "Median paired gain ± song-bootstrap 95% CI; same full inference, two seeds averaged\n"
             f"Fixed thresholds; all {songs} songs retained", fontsize=10, color=INK2, va="top")
    fig.savefig(OUT / "fig_activity_sensitivity.png")
    plt.close(fig)


def worst_figure(first: str, second: str, label: str, track: str) -> None:
    arrays = {}
    for config in (first, second):
        loaded = []
        for seed in (0, 1):
            rows = json.loads((OUT / "blocks" / f"{config}_seed{seed}.json").read_text())
            loaded.extend(row for row in rows if row["track"] == track)
        if len(loaded) != 2:
            raise ValueError(f"{config}: expected two seeds for {track}")
        arrays[config] = {key: np.mean([r[key] for r in loaded], axis=0)
                          for key in ("vocal_dbfs", "accompaniment_dbfs", "estimate_dbfs")}
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.11, hspace=0.42)
    t = np.arange(len(arrays[first]["vocal_dbfs"])) + .5
    axes[0].plot(t, arrays[first]["vocal_dbfs"], color=BLUE, label="Clean vocal")
    axes[0].plot(t, arrays[first]["accompaniment_dbfs"], color=GRAY, label="Accompaniment")
    axes[0].axhline(max(arrays[first]["vocal_dbfs"].max() - 40, -60), ls="--", color=INK2,
                    label="Primary activity threshold")
    axes[0].set_title("Reference source levels")
    for config, color in ((first, AQUA), (second, ORANGE)):
        axes[1].plot(t, arrays[config]["estimate_dbfs"], color=color, label=config.split("_")[0])
    axes[1].set_title("Model output levels (seed-average block dBFS)")
    for ax in axes:
        ax.set_ylabel("1 s RMS (dBFS)")
        ax.set_ylim(-100, 0)
        ax.grid(axis="y")
        # above the panel, right-aligned: the legend can no longer cover the curves
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=9, borderaxespad=0.2)
    axes[1].set_xlabel("Time within full song (s)")
    fig.text(0.02, 0.97, f"{track} · {label}", fontsize=15, fontweight="bold", color=INK, va="top")
    fig.savefig(OUT / f"fig_worst_{first.split('_')[0]}_{second.split('_')[0]}.png")
    plt.close(fig)


def main() -> None:
    provenance_path = OUT / "summary_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("partial"):
        raise SystemExit("refusing to replot a partial run")
    before = {name: sha(OUT / name) for name in FIGURES}
    paired = pd.read_csv(OUT / "paired.csv")
    cases = pd.read_csv(OUT / "cases.csv")
    songs = pd.read_csv(OUT / "per_track.csv").track.nunique()
    apply_style()
    activity_figure(paired, songs)
    for first, second, label in PAIRS:
        selected = cases[(cases.pair == label) & (cases.selection == "worst_1")]
        if len(selected) != 1:
            raise ValueError(f"expected one worst case for {label}")
        worst_figure(first, second, label, selected.iloc[0].track)
    after = {name: sha(OUT / name) for name in FIGURES}
    provenance["outputs"].update(after)
    provenance.setdefault("figure_replots", []).append({
        "date": date.today().isoformat(), "script": "scripts/replot_failure_figures.py",
        "script_sha256": sha(Path(__file__)),
        "reason": "legends moved outside the panels and header spacing fixed; drawn from the saved tables, "
                  "no inference and no change to any number",
        "inputs_sha256": {name: sha(OUT / name) for name in ("paired.csv", "cases.csv", "per_track.csv")},
        "figures_sha256_before": before, "figures_sha256_after": after})
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("redrawn:", ", ".join(FIGURES))


if __name__ == "__main__":
    main()
