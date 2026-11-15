"""Figures for DualWindow-MSS from saved results (no training or evaluation is re-run).

- fig_windows.png          method: analysis and synthesis windows of B and C
- fig_oracle_latency.png   oracle reference vs. algorithmic latency (results/oracle_validation.json)
- fig_quality_latency.png  test SI-SDR vs. algorithmic latency, with oracle references (results/eval/*_test.csv)
- fig_quality_cost.png     test SI-SDR vs. measured single-thread CPU cost (results/latency_benchmark.csv)
- fig_capacity.png         dual-window gain vs. GRU width, paired per song (results/capacity/*.csv from
                           scripts/summarize_capacity.py)
Figures whose inputs do not exist yet are skipped. The bleed figures come from scripts/bleed_benchmark.py.

Usage (from the repo root): python -I scripts/make_figures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dwmss.plotstyle import (AQUA, AXIS, BLUE, CONFIG_COLOR, CONFIG_LABEL, CONFIG_SHORT, GRAY, INK, INK2,  # noqa: E402
                             MUTED, ORANGE, SURFACE, WINDOW_TYPE, apply_style, dot, header, plt)

RES = REPO / "results"
CAP = RES / "capacity"
LAT = {"A_sym1024": 23.2, "B_sym256": 5.8, "C_dual": 5.8, "D_dual_2p9ms": 2.9, "E_sym128": 2.9}
SHIFT = {"short": 0.94, "dual": 1.06, "long": 1.0}  # draw same-latency configs side by side


def fig_windows():
    """Method figure: analysis and synthesis windows of B and C on a common time axis ending at 'now'."""
    from dwmss.stft import make_windows
    sr = 44100
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"hspace": 0.55})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.11)
    rows = [("B  short window", 256, 256, ORANGE), ("C  dual window", 2048, 256, AQUA)]
    for ax, (title, n, m, color) in zip(axes, rows):
        wa, ws = make_windows(n, m)
        t = (np.arange(n) - (n - 1)) / sr * 1000  # ms relative to the newest input sample
        lat = m / sr * 1000
        ax.axvspan(-lat, 0, color=GRAY, alpha=0.28, lw=0, zorder=0)
        ax.plot(t, wa, color=color, lw=2, zorder=3)
        if n != m:
            nz = ws > 0
            ax.plot(t[nz], ws[nz], color=color, lw=1.8, ls=(0, (3, 2)), zorder=3)
            ax.text(-lat - 2.2, 0.16, f"synthesis window (dashed): {m} samples ({lat:.1f} ms)", ha="right",
                    va="center", fontsize=9.5, color=INK)
            ax.annotate("", xy=(-lat * 0.80, 0.18), xytext=(-lat - 1.9, 0.16),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkA=0, shrinkB=2))
            ax.text(-n / sr * 1000, 1.18, f"analysis window: {n} samples ({n / sr * 1000:.1f} ms)  ·  "
                    f"frequency bins every {sr / n:.1f} Hz", ha="left", va="bottom", fontsize=9.5, color=INK)
        else:
            ax.text(-lat - 1.0, 0.62, f"analysis = synthesis window: {n} samples ({lat:.1f} ms)  ·  "
                    f"frequency bins every {sr / n:.0f} Hz", ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(-lat / 2, 1.42, f"latency\n{lat:.1f} ms", ha="center", va="top", fontsize=9, color=INK2,
                linespacing=1.05)
        ax.set_ylim(0, 1.45)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(-49, 1.5)
        ax.set_title(title, pad=6)
        ax.grid(True, axis="x")
    axes[1].set_xlabel("Time relative to the newest input sample (ms)")
    axes[1].set_xticks([-46.4, -40, -30, -20, -10, -5.8, 0], ["−46.4", "−40", "−30", "−20", "−10", "−5.8", "0"])
    header(fig, 0.07, 0.80, "Long analysis window, short synthesis window",
           "One STFT frame at 44.1 kHz, hop 128 samples (2.9 ms). The synthesis window sets the algorithmic latency "
           "(5.8 ms in both, shaded);\nthe analysis window sets the frequency resolution. Analysis × synthesis = "
           "Hann(256) in both, so overlap-add reconstructs the input exactly.")
    fig.savefig(RES / "fig_windows.png")
    plt.close(fig)


def fig_oracle_latency():
    f = RES / "oracle_validation.json"
    if not f.exists():
        return print("skip oracle figure (no results/oracle_validation.json)")
    s = pd.DataFrame(json.load(open(f))["summary"])
    sym = s[s["N"] == s["M"]].sort_values("latency_ms")
    dual = s[s["N"] != s["M"]]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.80, bottom=0.12)
    ax.plot(sym["latency_ms"], sym["irm2_median"], color=BLUE, lw=2, zorder=2)
    dot(ax, sym["latency_ms"], sym["irm2_median"], BLUE, size=60, label="Same window for analysis and synthesis")
    for _, r in sym.iterrows():
        ax.annotate(f"{r['N']}/{r['M']}", (r["latency_ms"], r["irm2_median"]), xytext=(10, -2),
                    textcoords="offset points", ha="left", va="center", fontsize=9, color=INK2)
    dot(ax, dual["latency_ms"], dual["irm2_median"], ORANGE, size=60, label="Dual window (long analysis / short synthesis)")
    for lat, g in dual.groupby("latency_ms"):
        best = g.loc[g["irm2_median"].idxmax()]
        for _, r in g.iterrows():
            ax.annotate(f"{r['N']}/{r['M']}", (r["latency_ms"], r["irm2_median"]), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK if r.name == best.name else INK2,
                        fontweight="bold" if r.name == best.name else "normal")
        base = sym[sym["latency_ms"] == lat]
        if len(base):
            y0 = float(base["irm2_median"].iloc[0])
            ax.annotate("", xy=(lat * 0.93, best["irm2_median"]), xytext=(lat * 0.93, y0),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
            ax.text(lat * 0.89, (y0 + best["irm2_median"]) / 2, f"+{best['irm2_median'] - y0:.1f} dB", ha="right",
                    va="center", fontsize=9.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sym["latency_ms"], [f"{v:.1f}" for v in sym["latency_ms"]])
    ax.set_xlim(sym["latency_ms"].min() / 1.5, sym["latency_ms"].max() * 1.45)
    ax.grid(True)
    ax.set_xlabel("Algorithmic latency (ms, log scale) — set by the synthesis window")
    ax.set_ylabel("Oracle vocal SDR (dB, median)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    header(fig, 0.01, 0.80, "A longer analysis window recovers quality at the same latency",
           f"Ideal power-ratio masks (an upper reference, not a trained model) on the "
           f"{len(json.load(open(f))['tracks'])} MUSDB18-HQ validation songs · 30 s vocal-rich excerpts, mono\n"
           "Labels = window lengths in samples at 44.1 kHz (analysis / synthesis)")
    fig.savefig(RES / "fig_oracle_latency.png")
    plt.close(fig)


def load_eval():
    frames = []
    for f in sorted((RES / "eval").glob("*_test.csv")):
        m = re.match(r"(.+)_seed(\d+)_test$", f.stem)
        if m and m.group(1) in LAT:
            d = pd.read_csv(f)
            d["config"], d["seed"] = m.group(1), int(m.group(2))
            frames.append(d)
    return pd.concat(frames) if frames else None


def fig_quality_latency(ev):
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fig.subplots_adjust(left=0.10, right=0.74, top=0.80, bottom=0.12)
    pts = {}
    for cfg, g in ev.groupby("config"):
        c, lat = CONFIG_COLOR[cfg], LAT[cfg]
        per_seed = g.groupby("seed")["si_sdr"].median()
        mean = g.groupby("track")["si_sdr"].mean().median()
        x = lat * SHIFT[WINDOW_TYPE[cfg]]
        pts[cfg] = (x, mean)
        ax.scatter([x] * len(per_seed), per_seed, s=22, color=c, alpha=0.45, zorder=2, lw=0)
        dot(ax, x, mean, c, size=80, zorder=3)
        if "oracle_irm2_si_sdr" in g:
            o = g.groupby("track")["oracle_irm2_si_sdr"].mean().median()
            ax.scatter(x, o, s=70, facecolors="none", edgecolors=c, linewidths=1.6, zorder=3)
    for cfg, (x, mean) in pts.items():
        # Label on the side away from the line that joins the same window type across latencies.
        same = [pts[c][0] for c in pts if c != cfg and WINDOW_TYPE[c] == WINDOW_TYPE[cfg]]
        left = bool(same) and all(s > x for s in same)
        ax.annotate(CONFIG_SHORT[cfg], (x, mean), xytext=(-12 if left else 10, 0), textcoords="offset points",
                    va="center", ha="right" if left else "left", fontsize=10.5, fontweight="bold", color=INK)
    for kind, color in (("short", ORANGE), ("dual", AQUA)):  # same window type across latencies
        xs = sorted((pts[c] for c in pts if WINDOW_TYPE[c] == kind), key=lambda p: p[0])
        if len(xs) > 1:
            ax.plot([p[0] for p in xs], [p[1] for p in xs], color=color, lw=1.6, alpha=0.6, zorder=1)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markersize=8, label="Long window (A)"),
               Line2D([], [], marker="o", ls="-", color=ORANGE, markersize=8, label="Short window (B 5.8 ms, E 2.9 ms)"),
               Line2D([], [], marker="o", ls="-", color=AQUA, markersize=8, label="Dual window (C 5.8 ms, D 2.9 ms)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK2, markersize=8,
                      label="Oracle mask, same windows"),
               Line2D([], [], marker="o", ls="", color=INK2, alpha=0.45, markersize=5, label="Individual seeds")]
    htd = RES / "eval" / "htdemucs_test.csv"
    if htd.exists():
        y = float(pd.read_csv(htd)["si_sdr"].median())
        handles.append(ax.axhline(y, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1, label="HTDemucs (offline, non-causal)"))
    ax.set_xscale("log", base=2)
    ax.set_xticks([2.9, 5.8, 11.6, 23.2], ["2.9", "5.8", "11.6", "23.2"])
    ax.set_xlim(2.0, 30)
    ax.grid(True)
    ax.set_xlabel("Algorithmic latency (ms, log scale)")
    ax.set_ylabel("Vocal SI-SDR on the test set (dB, median over songs)")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9.5)
    header(fig, 0.01, 0.80, "Separation quality vs. algorithmic latency",
           "Trained causal models on 50 MUSDB18-HQ test songs (full length, mono) · filled = 2 seeds averaged per song, "
           "small = single seeds\nhollow = oracle mask with the same windows · configs with the same latency are drawn "
           "side by side")
    fig.savefig(RES / "fig_quality_latency.png")
    plt.close(fig)


def fig_paired(ev):
    """Per-song SI-SDR difference between the dual and the short window at the same latency."""
    per = ev.groupby(["config", "track"])["si_sdr"].mean().unstack(0)
    pairs = [(a, b, lab) for a, b, lab in (("C_dual", "B_sym256", "C − B at 5.8 ms"),
                                           ("D_dual_2p9ms", "E_sym128", "D − E at 2.9 ms"))
             if a in per.columns and b in per.columns]
    if not pairs:
        return print("skip paired figure (needs B and C)")
    fig, axes = plt.subplots(len(pairs), 1, figsize=(9.6, 3.4 * len(pairs) + 1.25), squeeze=False, sharey=True)
    top = 1 - 1.25 / fig.get_size_inches()[1]
    fig.subplots_adjust(left=0.10, right=0.98, top=top, bottom=0.08, hspace=0.48)
    # One near-silent-vocal song (PR - Happy Daze, D - E about -20 dB) would otherwise set the shared scale and
    # flatten every other bar; the axis is clipped instead and the clipped value is printed next to its bar.
    floor = -3.2
    clipped = 0
    for ax, (a, b, lab) in zip(axes[:, 0], pairs):
        d = (per[a] - per[b]).dropna()
        # Match the per-comparison seed and track order in summarize_eval.py.
        rng = np.random.default_rng(0)
        boots = np.median(rng.choice(d.to_numpy(), size=(10000, len(d))), axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        d = d.sort_values()
        x = np.arange(len(d))
        ax.bar(x, d.clip(lower=floor), width=0.72, color=[AQUA if v > 0 else ORANGE for v in d], zorder=2)
        for i, v in zip(x, d):
            if v < floor:
                clipped += 1
                ax.annotate(f"{v:+.1f} dB, clipped".replace("-", "\u2212"), (i, floor), xytext=(9, 3), textcoords="offset points",
                            ha="left", va="bottom", fontsize=10, color=INK2)
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.axhspan(lo, hi, color=GRAY, alpha=0.35, lw=0, zorder=0)
        ax.axhline(float(d.median()), color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.text(0.97, 0.07, f"dual window better on {int((d > 0).sum())} of {len(d)} songs\n"
                f"median {d.median():+.2f} dB (95% CI {lo:+.2f} to {hi:+.2f})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11.5, color=INK, linespacing=1.35, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=4))
        ax.set_ylim(floor - 0.15, max(2.6, float(d.max()) + 0.3))
        ax.set_title(lab, fontsize=13, pad=12)
        ax.set_xticks([])
        ax.set_xlim(-1, len(d))
        ax.set_xlabel(f"{len(d)} test songs, sorted by the difference", fontsize=11.5)
        ax.set_ylabel("SI-SDR difference per song (dB)", fontsize=11.5)
        ax.tick_params(axis="y", labelsize=10.5, labelleft=True)
        ax.grid(True, axis="y")
        ax.spines["bottom"].set_visible(False)
    note = (f" · axis clipped at {floor:.1f} dB ({clipped} bar{'s' if clipped > 1 else ''})".replace("-", "\u2212")
            if clipped else "")
    header(fig, 0.01, top + 0.045, "The dual-window gain is broad, not driven by a few songs",
           "Per-song SI-SDR difference between the dual and the short window at the same algorithmic latency\n"
           f"(2 seeds averaged) · dashed: median · gray band: bootstrap 95% CI of the median{note}")
    fig.savefig(RES / "fig_paired_gain.png")
    plt.close(fig)


def fig_quality_cost(ev):
    f = RES / "latency_benchmark.csv"
    if not f.exists():
        return print("skip cost figure (no results/latency_benchmark.csv)")
    b = pd.read_csv(f)
    b = b[b["backend"] == "onnx"].set_index("config")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.80, bottom=0.12)
    pts = {}
    for cfg, g in ev.groupby("config"):
        if cfg not in b.index:
            continue
        pts[cfg] = (float(b.loc[cfg, "rtf"]), float(g.groupby("track")["si_sdr"].mean().median()))
    x_max = max(0.4, float(b["rtf"].max()) * 1.25)
    for cfg, (x, mean) in pts.items():
        # the 2.9 ms configs share their window type's color with the 5.8 ms ones: squares tell them apart
        dot(ax, x, mean, CONFIG_COLOR[cfg], size=80, label=f"{CONFIG_LABEL[cfg]}",
            marker="s" if cfg in {"D_dual_2p9ms", "E_sym128"} else "o")
        # A label drawn to the right would run into a point at nearly the same quality further right
        # (B and E both have 3.17 dB): put it below the point instead.
        blocked = any(0 < x2 - x < 0.3 and abs(y2 - mean) < 0.1 for c2, (x2, y2) in pts.items() if c2 != cfg)
        # near the right edge the label would leave the plot area: draw it to the left of the point
        leftward = not blocked and x > 0.6 * x_max
        ax.annotate(f"{CONFIG_SHORT[cfg]}  ·  P99 {b.loc[cfg, 'p99_us'] / 1000:.2f} ms per {b.loc[cfg, 'hop_ms']:.2f} ms hop",
                    (x, mean), xytext=(-6, -13) if blocked else ((-10, 0) if leftward else (10, 0)),
                    textcoords="offset points", va="top" if blocked else "center",
                    ha="right" if leftward else "left", fontsize=9.5, color=INK)
    ys = [p[1] for p in pts.values()]
    ax.set_ylim(min(ys) - 0.25, max(ys) + 0.15)
    ax.set_xlim(0, x_max)
    ax.grid(True)
    ax.set_xlabel("Real-time factor on one CPU thread (processing time ÷ audio time; < 1 = real time)")
    ax.set_ylabel("Vocal SI-SDR on the test set (dB)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    header(fig, 0.01, 0.80, "Quality vs. compute cost",
           "Full streaming pipeline per hop (STFT + model + iSTFT), ONNX Runtime, "
           f"{float((b['hops_timed'] * b['hop_ms']).median()) / 1000:.0f} s of streamed audio per model · "
           "CPU: Xeon Gold 6150, 1 thread")
    fig.savefig(RES / "fig_quality_cost.png")
    plt.close(fig)


def fig_capacity():
    sf, gf = CAP / "summary.csv", CAP / "gain.csv"
    if not (sf.exists() and gf.exists()):
        return print("skip capacity figure (no results/capacity/*.csv; run scripts/summarize_capacity.py)")
    try:
        s, g = pd.read_csv(sf), pd.read_csv(gf)
    except pd.errors.EmptyDataError:
        return print("skip capacity figure (empty results/capacity/*.csv)")
    need_g = {"params", "gain_median", "gain_median_ci_lo", "gain_median_ci_hi"}
    if len(g) < 2 or not need_g <= set(g.columns) or "rel_b64_median" not in s.columns:
        return print("skip capacity figure (fewer than two GRU widths with both B and C evaluated)")
    g = g.sort_values("params")
    fam = {"B": (ORANGE, "B  short window 256/256", 0.97), "C": (AQUA, "C  dual window 2048/256", 1.03)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.8), gridspec_kw={"wspace": 0.30})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.74, bottom=0.20)
    ticks = g["params"].tolist()
    tick_labels = [f"GRU {int(w)}\n{p / 1e6:.2f} M params" for w, p in zip(g["hidden"], g["params"])]
    lo_x, hi_x = min(ticks) / 1.3, max(ticks) * 1.3
    handles = []
    for key, (color, label, shift) in fam.items():
        r = s[s["family"] == key].sort_values("params")
        if r.empty:
            continue
        x = r["params"] * shift
        ax1.plot(x, r["rel_b64_median"], color=color, lw=2, zorder=2)
        ci = r.dropna(subset=["rel_b64_ci_lo"])
        ax1.vlines(ci["params"] * shift, ci["rel_b64_ci_lo"], ci["rel_b64_ci_hi"], color=color, lw=1.6, zorder=2)
        handles.append(dot(ax1, x, r["rel_b64_median"], color, size=70, zorder=3, label=label))
        ax1.annotate(key, (max(ticks) * fam["C"][2], float(r["rel_b64_median"].iloc[-1])), xytext=(10, 0),
                     textcoords="offset points", va="center", ha="left", fontsize=10.5, fontweight="bold", color=INK)
    ax1.axhline(0, color=AXIS, lw=1, zorder=1)
    ref = s[s["family"] == "A"]
    if len(ref):
        y = float(ref["rel_b64_median"].iloc[0])
        handles.append(ax1.axhline(y, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1,
                                   label="A  long window 1024/1024, 23.2 ms (GRU 256)"))
        ax1.annotate("A · 23.2 ms", (lo_x, y), xytext=(4, 4), textcoords="offset points", ha="left", va="bottom",
                     fontsize=9.5, color=INK2)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(ticks, tick_labels)
        ax.minorticks_off()
        ax.set_xlim(lo_x, hi_x)
        ax.grid(True, axis="y")
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0, y1 + 0.07 * (y1 - y0))  # headroom for the reference label
    ax1.set_ylabel("SI-SDR gain over B with GRU 64 (dB)")
    ax1.set_title("Quality relative to the smallest short-window model", pad=10)
    ax2.axhline(0, color=AXIS, lw=1, zorder=1)
    ax2.vlines(g["params"], g["gain_median_ci_lo"], g["gain_median_ci_hi"], color=INK2, lw=1.6, zorder=2)
    dot(ax2, g["params"], g["gain_median"], INK, size=60, zorder=3)
    for _, r in g.iterrows():
        ax2.annotate(f"{r['gain_median']:+.2f} dB", (r["params"], r["gain_median"]), xytext=(9, 0),
                     textcoords="offset points", va="center", ha="left", fontsize=10, color=INK)
    top = max(float(g["gain_median_ci_hi"].max()), 0.0)
    bot = min(float(g["gain_median_ci_lo"].min()), 0.0)
    pad = 0.15 * (top - bot if top > bot else 1.0)
    ax2.set_ylim(bot - pad, top + pad)
    ax2.set_xlim(lo_x, max(ticks) * 1.6)  # room for the value labels right of the last point
    ax2.set_ylabel("SI-SDR difference C − B (dB)")
    ax2.set_title("Gain from the dual window at each size", pad=10)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=3)
    n_songs = int(g["songs"].max())
    header(fig, 0.075, 0.80, "Dual-window gain at three model sizes",
           f"B and C differ only in the analysis window (both 5.8 ms latency); only the GRU width changes · "
           f"{n_songs} MUSDB18-HQ test songs, 2 seeds per model averaged per song\n"
           "Points = median of the per-song SI-SDR differences · bars = bootstrap 95% CI over songs · "
           "absolute values and means: results/capacity/summary.md")
    fig.savefig(RES / "fig_capacity.png")
    plt.close(fig)


def main():
    apply_style()
    fig_windows()
    fig_oracle_latency()
    ev = load_eval()
    if ev is None:
        print("skip test-set figures (no results/eval/*_test.csv yet)")
    else:
        fig_quality_latency(ev)
        fig_quality_cost(ev)
        fig_paired(ev)
    fig_capacity()


if __name__ == "__main__":
    main()
