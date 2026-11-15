"""Capacity experiment: does the dual-window STFT help small causal models more than large ones?

Background: Wu & Braun (ICASSP 2025, arXiv 2409.10358) found for speech enhancement that asymmetric
windows clearly help a weaker model but hardly help a strong one. Here the same question is asked for
vocal separation by varying only the GRU width (64 / 128 / 256) of B (256/256) and C (2048/256), both
5.8 ms. The width-256 runs are the main B_sym256 and C_dual runs.

Per width: paired per-song C − B difference of the seed-averaged SI-SDR with a bootstrap 95% CI over
songs, the spread over the 2 × 2 seed pairings, and the interaction (C − B at width w) − (C − B at 256),
also paired per song. The median of the per-song differences is the primary summary (a few songs have
almost no vocal and single runs differ by tens of dB on them, which dominates the mean); means are
reported as well. Also: the effect of the GRU width within each window (paired, 64 → 256), and every
config relative to the smallest short-window model (B, GRU 64), which the figure uses. Parameters and
MACs come from the model definition (dwmss/complexity.py); CPU timings come from
results/latency_benchmark.csv (slurm/benchmark.sbatch) when it exists.

Usage: python -I scripts/summarize_capacity.py
Outputs: results/capacity/summary.md, summary.csv (one row per config), gain.csv (one row per width)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dwmss.complexity import describe  # noqa: E402
from dwmss.config import load_config  # noqa: E402

FAMILIES = {"B": "B_sym256", "C": "C_dual"}
WIDTHS = [64, 128, 256]
REFERENCE = "A_sym1024"
EVAL = REPO / "results" / "eval"
OUT = REPO / "results" / "capacity"
BENCH = REPO / "results" / "latency_benchmark.csv"


def config_name(family: str, width: int) -> str:
    base = FAMILIES[family]
    return base if width == 256 else f"{base}_h{width}"


def boot(x, stat=np.mean, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = stat(rng.choice(x, size=(n, len(x))), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Per-song a − b: median and mean with bootstrap 95% CIs, share of songs where a is better."""
    d = (a - b).dropna()
    (mlo, mhi), (lo, hi) = boot(d, np.median), boot(d)
    return {"n": len(d), "median": float(d.median()), "median_lo": mlo, "median_hi": mhi,
            "mean": float(d.mean()), "mean_lo": lo, "mean_hi": hi, "frac_better": float((d > 0).mean())}


def fmt(p: dict, stat: str) -> str:
    return f"{p[stat]:+.2f} [{p[stat + '_lo']:+.2f}, {p[stat + '_hi']:+.2f}]"


def load_runs(cfg: str) -> pd.DataFrame | None:
    frames = []
    for f in sorted(EVAL.glob(f"{cfg}_seed*_test.csv")):
        seed = f.stem[len(cfg) + len("_seed"):-len("_test")]
        if seed.isdigit():
            d = pd.read_csv(f)
            d["seed"] = int(seed)
            frames.append(d)
    return pd.concat(frames) if frames else None


def main():
    bench = None
    if BENCH.exists():
        bench = pd.read_csv(BENCH)
        bench = bench[bench["backend"] == "onnx"].set_index("config")

    rows, per_track, per_seed_track, missing = [], {}, {}, []
    for fam, w in [("A", 256)] + list(itertools.product(FAMILIES, WIDTHS)):
        cfg = REFERENCE if fam == "A" else config_name(fam, w)
        d = load_runs(cfg)
        if d is None:
            missing.append(cfg)
            continue
        info = describe(load_config(str(REPO / "configs" / f"{cfg}.yaml")))
        seeds = sorted(d["seed"].unique())
        per_track[cfg] = d.groupby("track")["si_sdr"].mean()
        per_seed_track[cfg] = {s: d[d.seed == s].set_index("track")["si_sdr"] for s in seeds}
        row = {"config": cfg, "family": fam, "hidden": w, **info, "seeds": len(seeds),
               "si_sdr_median": float(per_track[cfg].median()),
               "si_sdri_median": float(d.groupby("track")["si_sdri"].mean().median())}
        for s in seeds:
            row[f"si_sdr_seed{s}"] = float(per_seed_track[cfg][s].median())
        if bench is not None and cfg in bench.index:
            for k in ("rtf", "p50_us", "p99_us", "deadline_miss_rate"):
                row[k] = float(bench.loc[cfg, k])
        rows.append(row)
    if not any(r["family"] in FAMILIES for r in rows):
        print("no capacity evaluations yet (results/eval/{B_sym256,C_dual}*_test.csv); nothing written")
        return
    summary = pd.DataFrame(rows)
    ref = config_name("B", WIDTHS[0])
    if ref in per_track:
        for i, r in summary.iterrows():
            if r["config"] == ref:
                summary.loc[i, ["rel_b64_median", "rel_b64_ci_lo", "rel_b64_ci_hi"]] = 0.0, np.nan, np.nan
                continue
            q = paired(per_track[r["config"]], per_track[ref])
            summary.loc[i, ["rel_b64_median", "rel_b64_ci_lo", "rel_b64_ci_hi"]] = (q["median"], q["median_lo"],
                                                                                  q["median_hi"])

    gains = []
    base_c, base_b = config_name("C", 256), config_name("B", 256)
    for w in WIDTHS:
        c, b = config_name("C", w), config_name("B", w)
        if c not in per_track or b not in per_track:
            continue
        q = paired(per_track[c], per_track[b])
        pairs = [float((per_seed_track[c][sc] - per_seed_track[b][sb]).dropna().median())
                 for sc in per_seed_track[c] for sb in per_seed_track[b]]
        g = {"hidden": w, "params": int(summary.loc[summary.config == c, "params"].iloc[0]), "songs": q["n"],
             "gain_median": q["median"], "gain_median_ci_lo": q["median_lo"], "gain_median_ci_hi": q["median_hi"],
             "gain_mean": q["mean"], "gain_ci_lo": q["mean_lo"], "gain_ci_hi": q["mean_hi"],
             "frac_songs_c_better": q["frac_better"], "seed_pairs": len(pairs),
             "gain_seedpair_min": min(pairs), "gain_seedpair_max": max(pairs)}
        if w != 256 and base_c in per_track and base_b in per_track:
            qi = paired(per_track[c] - per_track[b], per_track[base_c] - per_track[base_b])
            g.update({"interaction_vs256_median": qi["median"], "interaction_vs256_median_ci_lo": qi["median_lo"],
                      "interaction_vs256_median_ci_hi": qi["median_hi"], "interaction_vs256_mean": qi["mean"],
                      "interaction_vs256_ci_lo": qi["mean_lo"], "interaction_vs256_ci_hi": qi["mean_hi"]})
        gains.append(g)
    size_effects = []  # (label, a, b): paired a − b
    for fam in FAMILIES:
        a, b = config_name(fam, WIDTHS[-1]), config_name(fam, WIDTHS[0])
        if a in per_track and b in per_track:
            size_effects.append((f"{fam}: GRU {WIDTHS[-1]} − GRU {WIDTHS[0]}", a, b))
    a, b = config_name("C", WIDTHS[0]), config_name("B", WIDTHS[-1])
    if a in per_track and b in per_track:
        size_effects.append((f"C at GRU {WIDTHS[0]} − B at GRU {WIDTHS[-1]}", a, b))
    gain = pd.DataFrame(gains)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    gain.to_csv(OUT / "gain.csv", index=False)

    L = ["# Capacity experiment: dual-window gain vs. model size", "",
         "B = 256/256 and C = 2048/256 (both 5.8 ms algorithmic latency); only the GRU width changes. "
         "MUSDB18-HQ test set (full songs, mono, vocals). SI-SDR in dB, median over songs of the seed-averaged "
         "value. A (1024/1024, 23.2 ms, GRU 256) is listed as the long-window reference.", ""]
    if missing:
        L += [f"Missing evaluations (not included): {', '.join(missing)}", ""]
    L += ["| config | GRU width | params | MMAC / frame | GMAC / s | seeds | SI-SDR per seed | SI-SDR | SI-SDRi | "
          "RTF (1 thread) | P99 per hop (µs) |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        per_seed = " / ".join(f"{r[k]:.2f}" for k in sorted(c for c in summary.columns if c.startswith("si_sdr_seed"))
                              if pd.notna(r.get(k)))
        rtf = f"{r['rtf']:.3f}" if "rtf" in r and pd.notna(r.get("rtf")) else "–"
        p99 = f"{r['p99_us']:.0f}" if "p99_us" in r and pd.notna(r.get("p99_us")) else "–"
        L.append(f"| {r['config']} | {r['hidden']} | {r['params']:,} | {r['macs_per_frame_M']:.2f} | "
                 f"{r['gmacs_per_s']:.2f} | {r['seeds']} | {per_seed} | {r['si_sdr_median']:.2f} | "
                 f"{r['si_sdri_median']:.2f} | {rtf} | {p99} |")
    L += ["", "## Dual-window gain C − B per width (paired per song, seed-averaged)", "",
          "| GRU width | params | songs | median gain [95% CI] | mean gain [95% CI] | songs where C is better | "
          "median gain over seed pairings | gain minus gain at width 256: median [95% CI] / mean [95% CI] |",
          "|---:|---:|---:|---|---|---:|---|---|"]
    for _, g in gain.iterrows():
        if pd.notna(g.get("interaction_vs256_median")):
            inter = (f"{g['interaction_vs256_median']:+.2f} [{g['interaction_vs256_median_ci_lo']:+.2f}, "
                     f"{g['interaction_vs256_median_ci_hi']:+.2f}] / {g['interaction_vs256_mean']:+.2f} "
                     f"[{g['interaction_vs256_ci_lo']:+.2f}, {g['interaction_vs256_ci_hi']:+.2f}]")
        else:
            inter = "–"
        L.append(f"| {int(g['hidden'])} | {int(g['params']):,} | {int(g['songs'])} | {g['gain_median']:+.2f} "
                 f"[{g['gain_median_ci_lo']:+.2f}, {g['gain_median_ci_hi']:+.2f}] | {g['gain_mean']:+.2f} "
                 f"[{g['gain_ci_lo']:+.2f}, {g['gain_ci_hi']:+.2f}] | {g['frac_songs_c_better']:.0%} | "
                 f"{g['gain_seedpair_min']:+.2f} … {g['gain_seedpair_max']:+.2f} ({int(g['seed_pairs'])}) | {inter} |")
    if size_effects:
        L += ["", "## Effect of model size (paired per song, seed-averaged)", "",
              "| comparison | params | MMAC / frame | median difference [95% CI] | mean difference [95% CI] | "
              "songs where the first is better |", "|---|---|---|---|---|---:|"]
        info = summary.set_index("config")
        for label, a, b in size_effects:
            q = paired(per_track[a], per_track[b])
            L.append(f"| {label} | {info.loc[a, 'params'] / 1e6:.2f} M vs. {info.loc[b, 'params'] / 1e6:.2f} M | "
                     f"{info.loc[a, 'macs_per_frame_M']:.2f} vs. {info.loc[b, 'macs_per_frame_M']:.2f} | "
                     f"{fmt(q, 'median')} | {fmt(q, 'mean')} | {q['frac_better']:.0%} |")
    mix = pd.concat([load_runs(c) for c in per_track]).groupby("track")["si_sdr_mixture"].mean()
    t = mix.idxmin()
    on_t = pd.concat([load_runs(c) for c in per_track if c != REFERENCE])
    on_t = on_t[on_t.track == t]["si_sdr"]
    L += ["", "Bootstrap CIs resample songs (10,000 resamples) and do not include training-seed variance; the "
              "seed-pairing column shows that variance separately (median gain for each of the 2 × 2 pairings of "
              "seeds). A positive last column means the dual window helps the smaller model more than the "
              "width-256 model.",
          "", "The median is the primary summary. A few songs have almost no vocal: on "
              f"{t} (mixture SI-SDR {mix[t]:.1f} dB, test-set median {mix.median():.1f} dB) the single B/C runs "
              f"score {on_t.max():.0f} to {on_t.min():.0f} dB, so one run can move the mean over 50 songs by "
              "several tenths of a dB.",
          "", "Compute note: the frequency-direction encoder and the per-bin head run on every FFT bin, so C "
              "(1025 bins) costs more per frame than B (129 bins) at the same width; the GRU width mainly changes "
              "the parameter count. MACs are counted from the layer shapes (dwmss/complexity.py)."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
