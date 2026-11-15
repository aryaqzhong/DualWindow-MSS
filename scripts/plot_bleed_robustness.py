"""Plot saved frozen-model bleed-path results; never run inference or modify its protocol.

python -I scripts/plot_bleed_robustness.py
python -I scripts/plot_bleed_robustness.py --results /tmp/bleed-smoke --allow-partial

Production requires all 50 songs and both-seed summaries for B/C/D/E. Partial
plots are allowed only in isolated directories and are visibly marked as smoke
tests. Confidence intervals come directly from paired_comparisons.csv.
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
from matplotlib.lines import Line2D  # noqa: E402

from dwmss.plotstyle import (AXIS, CONFIG_COLOR, CONFIG_SHORT, INK, INK2, SURFACE,  # noqa: E402
                             apply_style, header, plt)

PRODUCTION = REPO / "results/bleed_robustness"
CONFIGS = ("B_sym256", "C_dual", "E_sym128", "D_dual_2p9ms")
SCENARIOS = ("dry", "filtered_delay", "synthetic_room")
GAMMAS = {0.5, 0.75, 1.0, 1.5, 2.0}
PATH_LABEL = {"dry": "Dry control", "filtered_delay": "Filter + delay", "synthetic_room": "Synthetic room"}
LATENCY = {"B_sym256": 5.8, "C_dual": 5.8, "E_sym128": 2.9, "D_dual_2p9ms": 2.9}
WINDOW = {"B_sym256": "short", "C_dual": "dual", "E_sym128": "short", "D_dual_2p9ms": "dual"}
PANELS = ((5.8, ("B_sym256", "C_dual")), (2.9, ("E_sym128", "D_dual_2p9ms")))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as f:
            temp = f.name
            f.write(text)
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


def save_figure(fig, path, metadata):
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".png")
    os.close(fd)
    try:
        fig.savefig(temp, format="png", metadata={"Description": json.dumps(metadata, sort_keys=True)})
        os.replace(temp, path)
    finally:
        plt.close(fig)
        if os.path.exists(temp):
            os.unlink(temp)


def load_results(results, allow_partial=False):
    per_song = pd.read_csv(results / "per_song.csv")
    comparisons = pd.read_csv(results / "paired_comparisons.csv")
    provenance = json.loads((results / "summary_provenance.json").read_text())
    required = {"config", "song", "scenario", "gamma", "si_sdr"}
    if not required <= set(per_song):
        raise ValueError("per_song.csv lacks required columns")
    expected_comparison = {"kind", "comparison", "scenario", "metric", "songs", "median",
                           "median_ci_low", "median_ci_high", "mean"}
    if not expected_comparison <= set(comparisons):
        raise ValueError("paired_comparisons.csv lacks required columns")
    if per_song.empty or per_song.duplicated(["config", "song", "scenario", "gamma"]).any():
        raise ValueError("per-song curves are empty or have duplicate cells")
    configs = set(per_song.config)
    if not configs <= set(CONFIGS) or configs != set(provenance["included_configs"]):
        raise ValueError("config coverage does not match summary provenance")
    song_sets = [set(g.song) for _, g in per_song.groupby("config")]
    songs = song_sets[0]
    if not songs or any(s != songs for s in song_sets) or len(songs) != provenance["songs"]:
        raise ValueError("configs have different song sets or disagree with provenance")
    if not provenance.get("protocol_id") or not provenance.get("excerpt_manifest_id"):
        raise ValueError("summary provenance lacks protocol or excerpt identity")
    if not np.isfinite(per_song[["gamma", "si_sdr"]].to_numpy()).all():
        raise ValueError("nonfinite per-song data")
    if "actual_bvr_db" in per_song and not np.allclose(per_song.actual_bvr_db, -5.0, atol=1e-4):
        raise ValueError("unexpected BVR; this figure is specified for -5 dB")
    for _, group in per_song.groupby(["config", "song"]):
        coverage = set(zip(group.scenario, group.gamma))
        if coverage != {(s, g) for s in SCENARIOS for g in GAMMAS}:
            raise ValueError("incomplete scenario/gamma coverage in a song")
    complete = configs == set(CONFIGS) and len(songs) == 50 and not provenance.get("incomplete_runs")
    if not complete and not allow_partial:
        raise ValueError("production figures require all B/C/D/E, 50 songs and no incomplete runs; "
                         "use --allow-partial only with an isolated smoke-test directory")
    main = per_song[per_song.gamma == 1.0].copy()
    changes = comparisons[(comparisons.kind == "path_minus_dry") & (comparisons.metric == "si_sdr")].copy()
    expected = {(c, s) for c in configs for s in SCENARIOS[1:]}
    if changes.duplicated(["comparison", "scenario"]).any() or set(zip(changes.comparison, changes.scenario)) != expected:
        raise ValueError("paired path-minus-dry comparisons are missing or duplicated")
    if not np.isfinite(changes[["songs", "median", "median_ci_low", "median_ci_high", "mean"]].to_numpy()).all():
        raise ValueError("nonfinite paired comparison or confidence interval")
    for _, row in changes.iterrows():
        wide = main[main.config == row.comparison].pivot(index="song", columns="scenario", values="si_sdr")
        delta = wide[row.scenario] - wide.dry
        if (row.songs != len(delta) or row.median_ci_low > row.median_ci_high
                or not np.isclose(row["median"], delta.median(), atol=1e-8, rtol=1e-8)
                or not np.isclose(row["mean"], delta.mean(), atol=1e-8, rtol=1e-8)):
            raise ValueError("saved paired comparison disagrees with the per-song results")
    return main, changes, provenance, not complete


def subtitle(provenance, partial):
    prefix = "SMOKE TEST ONLY · " if partial else ""
    songs = provenance["songs"]
    return (f"{prefix}{songs} MUSDB18-HQ test song{'' if songs == 1 else 's'} · 2 frozen training seeds averaged per song · "
            "30 s scored + 5 s context\n"
            "Clean vocal, transformed accompaniment re-normalized to −5 dB · mask strength γ = 1")


def plot_absolute(main, provenance, partial, out):
    configs = set(main.config)
    panels = [(lat, [c for c in cs if c in configs]) for lat, cs in PANELS if any(c in configs for c in cs)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels) + 0.6, 5.8),
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.75, bottom=0.19, wspace=0.16)
    for ax, (latency, panel_configs) in zip(axes[0], panels):
        handles = []
        for i, config in enumerate(panel_configs):
            values = main[main.config == config].groupby("scenario").si_sdr.median().reindex(SCENARIOS)
            marker = "o" if WINDOW[config] == "short" else "s"
            line, = ax.plot(range(3), values, color=CONFIG_COLOR[config], marker=marker, markersize=7,
                            markeredgecolor=SURFACE, markeredgewidth=1.4, linewidth=1.8,
                            label=f"{CONFIG_SHORT[config]} · {WINDOW[config]} window")
            handles.append(line)
            ax.annotate(f"{CONFIG_SHORT[config]}  {values.iloc[-1]:.2f}", (2, values.iloc[-1]),
                        xytext=(8, -8 if i == 0 else 8), textcoords="offset points",
                        ha="left", va="center", color=INK, fontsize=10)
        ax.set_title(f"{latency:g} ms algorithmic latency", pad=12)
        ax.set_xticks(range(3), ["Dry\ncontrol", "Filter +\ndelay", "Synthetic\nroom"])
        ax.set_xlim(-0.18, 2.67)
        ax.grid(True, axis="y")
        ax.margins(y=0.18)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, -0.29), ncol=2)
    axes[0][0].set_ylabel("Vocal SI-SDR (dB, median over songs)")
    header(fig, 0.09, 0.81, "Vocal separation under fixed accompaniment paths", subtitle(provenance, partial))
    fig.text(0.09, 0.015, "Synthetic feed-forward FIRs; not real microphones or acoustic feedback. "
             "Paired changes are shown in the companion figure.", color=INK2, fontsize=9)
    save_figure(fig, out / "fig_bleed_paths_quality.png",
                {"protocol_id": provenance["protocol_id"], "partial_smoke": partial,
                 "statistic": "median over songs of two-seed-averaged gamma=1 SI-SDR"})


def plot_changes(changes, provenance, partial, out):
    configs = [c for c in CONFIGS if c in set(changes.comparison)]
    positions = {c: i + (0.55 if LATENCY[c] == 2.9 and any(LATENCY[x] == 5.8 for x in configs) else 0)
                 for i, c in enumerate(configs)}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.75, bottom=0.19, wspace=0.14)
    extent = np.r_[changes.median_ci_low, changes.median_ci_high, 0.0]
    span = max(float(np.ptp(extent)), 0.2)
    xlim = (float(extent.min()) - 0.15 * span, float(extent.max()) + 0.28 * span)
    for ax, scenario in zip(axes, SCENARIOS[1:]):
        group = changes[changes.scenario == scenario].set_index("comparison")
        for config in configs:
            row, y = group.loc[config], positions[config]
            color = CONFIG_COLOR[config]
            marker = "o" if LATENCY[config] == 5.8 else "s"
            ax.hlines(y, row.median_ci_low, row.median_ci_high, color=color, linewidth=2, zorder=2)
            ax.vlines([row.median_ci_low, row.median_ci_high], y - 0.055, y + 0.055, color=color, linewidth=1)
            ax.scatter(row["median"], y, color=color, marker=marker, s=65,
                       edgecolors=SURFACE, linewidths=1.4, zorder=3)
            ax.annotate(f"{row['median']:+.2f}", (row["median"], y), xytext=(7, 7),
                        textcoords="offset points", fontsize=9.5, color=INK, ha="left", va="bottom")
        ax.axvline(0, color=AXIS, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(PATH_LABEL[scenario], pad=12)
        ax.set_xlim(*xlim)
        ax.grid(True, axis="x")
        ax.set_yticks([positions[c] for c in configs],
                      [f"{CONFIG_SHORT[c]} · {WINDOW[c]} · {LATENCY[c]:g} ms" for c in configs])
        ax.set_ylim(max(positions.values()) + 0.55, -0.65)
        ax.set_xlabel("SI-SDR change from dry (dB)")
    header(fig, 0.09, 0.81, "Paired changes from the dry control",
           subtitle(provenance, partial) + "\nPoints: median of per-song changes · bars: bootstrap 95% CI over songs · negative = worse")
    handles = [Line2D([], [], color=INK2, marker="o", linestyle="", label="5.8 ms", markersize=6),
               Line2D([], [], color=INK2, marker="s", linestyle="", label="2.9 ms", markersize=6)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.57, 0.065), ncol=2)
    fig.text(0.09, 0.015, "Synthetic feed-forward FIRs; not real microphones or acoustic feedback. "
             "Confidence intervals exclude training-seed uncertainty.", color=INK2, fontsize=9)
    save_figure(fig, out / "fig_bleed_paths_change.png",
                {"protocol_id": provenance["protocol_id"], "partial_smoke": partial,
                 "statistic": "paired path-minus-dry median SI-SDR change and saved song-bootstrap 95% CI"})


def worst_changes(main, top_k):
    rows = []
    for config in CONFIGS:
        group = main[main.config == config]
        if group.empty:
            continue
        wide = group.pivot(index="song", columns="scenario", values="si_sdr")
        candidates = []
        for scenario in SCENARIOS[1:]:
            for song, r in wide.iterrows():
                candidates.append({"config": config, "latency_ms": LATENCY[config], "scenario": scenario,
                                   "song": song, "dry_si_sdr": r.dry, "path_si_sdr": r[scenario],
                                   "path_minus_dry_db": r[scenario] - r.dry})
        ranked = pd.DataFrame(candidates).sort_values(["path_minus_dry_db", "song", "scenario"], kind="stable").head(top_k)
        ranked.insert(0, "rank_within_model", range(1, len(ranked) + 1))
        rows.extend(ranked.to_dict("records"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=PRODUCTION)
    ap.add_argument("--out", type=Path, help="figure directory (default: --results)")
    ap.add_argument("--allow-partial", action="store_true", help="visibly marked smoke figures in isolated directories only")
    ap.add_argument("--top-failures", type=int, default=5,
                    help="number of smallest path-minus-dry changes per model to export (may include positive changes)")
    args = ap.parse_args()
    out = args.out or args.results
    if args.top_failures < 1:
        ap.error("--top-failures must be positive")
    if args.allow_partial and (args.results.resolve() == PRODUCTION.resolve() or out.resolve() == PRODUCTION.resolve()):
        ap.error("--allow-partial requires isolated input and output directories, outside production results")
    try:
        main_rows, changes, provenance, partial = load_results(args.results, args.allow_partial)
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as error:
        ap.error(str(error))
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    plot_absolute(main_rows, provenance, partial, out)
    plot_changes(changes, provenance, partial, out)
    cases = worst_changes(main_rows, args.top_failures)
    atomic_text(out / "worst_path_changes.csv", cases.to_csv(index=False))
    artifacts = ("fig_bleed_paths_quality.png", "fig_bleed_paths_change.png", "worst_path_changes.csv")
    record = {"protocol_id": provenance["protocol_id"], "excerpt_manifest_id": provenance["excerpt_manifest_id"],
              "partial_smoke": partial, "songs": provenance["songs"],
              "configs": [c for c in CONFIGS if c in set(main_rows.config)],
              "inputs_sha256": {n: sha256(args.results / n) for n in
                                ("per_song.csv", "paired_comparisons.csv", "summary_provenance.json")},
              "plotter_sha256": sha256(Path(__file__)),
              "artifacts_sha256": {n: sha256(out / n) for n in artifacts},
              "case_selection": f"Smallest {args.top_failures} path-minus-dry SI-SDR changes per model, across both transformed paths; ties by song then path. Descriptive post-hoc selection, not held-out evaluation. Positive changes are improvements, even if ranked here.",
              "ci_source": "Saved paired song-bootstrap intervals; no recomputation or training-seed uncertainty."}
    atomic_text(out / "figures_provenance.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {', '.join(artifacts)} and figures_provenance.json in {out}")


if __name__ == "__main__":
    main()
