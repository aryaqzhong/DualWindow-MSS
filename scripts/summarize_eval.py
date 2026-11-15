"""Summarize test-set evaluations (results/eval/*_test.csv) into results/summary_test.md.

Per config: SI-SDR / SI-SDRi per seed and averaged over seeds, oracle IRM2 reference.
Paired per-track comparisons (seed-averaged): C − B (same 5.8 ms latency; the main question) and
A − C (quality cost of going from 23.2 ms to 5.8 ms), with bootstrap 95% CIs over tracks. The median of
the per-track differences is the primary summary: a few tracks have almost no vocal, every model scores far
below 0 dB on them and single runs differ by tens of dB there, which dominates the mean (reported too).

Usage: python -I scripts/summarize_eval.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LATENCY = {"A_sym1024": 23.2, "B_sym256": 5.8, "C_dual": 5.8, "D_dual_2p9ms": 2.9, "E_sym128": 2.9}


def boot(x, stat=np.mean, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = stat(rng.choice(x, size=(n, len(x))), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ev = REPO / "results" / "eval"
    frames = []
    for f in sorted(ev.glob("*_test.csv")):
        m = re.match(r"(.+)_seed(\d+)_test$", f.stem)
        if not m or m.group(1) not in LATENCY:  # capacity runs are summarized by summarize_capacity.py
            continue
        d = pd.read_csv(f)
        d["config"], d["seed"] = m.group(1), int(m.group(2))
        frames.append(d)
    if not frames:
        print("no evaluation files found")
        return
    df = pd.concat(frames)
    L = ["# Test-set results (MUSDB18-HQ, 50 songs, full length, mono, vocals)", "",
         "SI-SDR in dB, median over songs. Oracle = IRM2 with the same window configuration (non-deployable "
         "reference).", "",
         "| config | latency (ms) | seeds | SI-SDR per seed | SI-SDR (seed mean) | SI-SDRi (seed mean) | oracle IRM2 SI-SDR |",
         "|---|---:|---:|---|---:|---:|---:|"]
    per_track = {}
    for cfg, g in df.groupby("config"):
        seeds = sorted(g["seed"].unique())
        per_seed = [g[g.seed == s]["si_sdr"].median() for s in seeds]
        avg = g.groupby("track")[["si_sdr", "si_sdri"]].mean()
        per_track[cfg] = avg["si_sdr"]
        oracle = g.groupby("track")["oracle_irm2_si_sdr"].mean().median() if "oracle_irm2_si_sdr" in g else np.nan
        L.append(f"| {cfg} | {LATENCY.get(cfg, float('nan')):.1f} | {len(seeds)} | "
                 f"{' / '.join(f'{v:.2f}' for v in per_seed)} | {avg['si_sdr'].median():.2f} | "
                 f"{avg['si_sdri'].median():.2f} | {oracle:.2f} |")
    L += ["", "## Paired comparisons (per song, seed-averaged SI-SDR)", "",
          "| comparison | songs | median difference [95% CI] | mean difference [95% CI] | songs where first is better |",
          "|---|---:|---|---|---:|"]
    for a, b, label in (("C_dual", "B_sym256", "C − B (dual vs. short window, both 5.8 ms)"),
                        ("A_sym1024", "C_dual", "A − C (23.2 ms vs. 5.8 ms dual)"),
                        ("A_sym1024", "B_sym256", "A − B (23.2 ms vs. 5.8 ms short)"),
                        ("D_dual_2p9ms", "E_sym128", "D − E (dual vs. short window, both 2.9 ms)"),
                        ("C_dual", "D_dual_2p9ms", "C − D (5.8 ms vs. 2.9 ms dual)"),
                        ("B_sym256", "E_sym128", "B − E (5.8 ms vs. 2.9 ms short)")):
        if a in per_track and b in per_track:
            d = (per_track[a] - per_track[b]).dropna()
            lo, hi = boot(d)
            mlo, mhi = boot(d, np.median)
            L.append(f"| {label} | {len(d)} | {d.median():+.2f} [{mlo:+.2f}, {mhi:+.2f}] | "
                     f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {(d > 0).mean():.0%} |")
    mix = df.groupby("track")["si_sdr_mixture"].mean()
    t = mix.idxmin()
    on_t = df[df.track == t]["si_sdr"]
    L += ["", "Bootstrap CIs resample songs (10,000 resamples). The median is the primary summary because a few "
              f"songs have almost no vocal: on {t} (mixture SI-SDR {mix[t]:.1f} dB, test-set median "
              f"{mix.median():.1f} dB) the single runs score {on_t.max():.0f} to {on_t.min():.0f} dB, and such songs "
              "dominate the mean."]
    htd = ev / "htdemucs_test.csv"
    if htd.exists() and "A_sym1024" in per_track:
        h = pd.read_csv(htd).set_index("track")
        d = (h["si_sdr"] - per_track["A_sym1024"]).dropna()
        mlo, mhi = boot(d, np.median)
        L += ["", "## Offline reference", "",
              f"HTDemucs (pretrained, non-causal, ~7.8 s segments; same mono input duplicated to stereo): SI-SDR "
              f"{h['si_sdr'].median():.2f} dB, SI-SDRi {h['si_sdri'].median():.2f} dB (medians over {len(h)} songs). "
              f"Per song it is {d.median():+.2f} dB [{mlo:+.2f}, {mhi:+.2f}] above A (23.2 ms). It has no low-latency "
              "operating point and was trained on more data (`scripts/htdemucs_reference.py`)."]
    out = REPO / "results" / "summary_test.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
