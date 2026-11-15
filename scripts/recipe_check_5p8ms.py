"""Exploratory 5.8 ms learning-rate check for the wide short-window control.

Design: results/recipe_check_5p8ms/design.md.
  --select     pick the wide-B learning rate from validation logs only; writes selection.json and prints
               the selected config name (last stdout line)
  --summarize  paired test comparisons for the selected setting; writes summary.md and paired.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "recipe_check_5p8ms"
EVAL = REPO / "results" / "eval"
CANDIDATES = {1e-3: "B_sym256_compute", 3e-4: "B_sym256_compute_lr3e4"}
SEEDS = (0, 1)
STEPS = 200_000
TIE_DB = 1e-9
N_BOOT = 10_000


def best_validation(run: Path) -> dict:
    """Best validation median SI-SDR of a finished run (first maximum on ties)."""
    if not (run / "log.jsonl").exists():
        raise SystemExit(f"{run.name}: no training log; refusing to select")
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()
            if '"val_si_sdr_median"' in line]
    if not rows or not (run / "DONE").exists() or max(r["step"] for r in rows) < STEPS:
        raise SystemExit(f"{run.name}: training not finished; refusing to select")
    best = rows[0]
    for r in rows[1:]:
        if r["val_si_sdr_median"] > best["val_si_sdr_median"]:
            best = r
    return {"best_step": best["step"], "best_val_si_sdr_median": best["val_si_sdr_median"]}


def select() -> str:
    table = {}
    for lr, name in CANDIDATES.items():
        per_seed = {s: best_validation(REPO / "runs" / f"{name}_seed{s}") for s in SEEDS}
        mean = float(np.mean([v["best_val_si_sdr_median"] for v in per_seed.values()]))
        table[name] = {"lr": lr, "seeds": per_seed, "mean_best_val": mean}
    base, alt = CANDIDATES[1e-3], CANDIDATES[3e-4]
    chosen = alt if table[alt]["mean_best_val"] > table[base]["mean_best_val"] + TIE_DB else base
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(
        {"rule": "mean over seeds of each seed's best validation median SI-SDR; ties within 1e-9 dB keep 1e-3",
         "candidates": table, "selected": chosen}, indent=2))
    for name, v in table.items():
        print(f"{name}: mean best validation {v['mean_best_val']:.3f} dB", file=sys.stderr)
    print(chosen)
    return chosen


def per_seed(name: str) -> list[pd.Series]:
    return [pd.read_csv(EVAL / f"{name}_seed{s}_test.csv").set_index("track")["si_sdr"] for s in SEEDS]


def paired(first: str, second: str) -> dict:
    a, b = per_seed(first), per_seed(second)
    d = ((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2).dropna()
    x = d.to_numpy()
    rng = np.random.default_rng(0)
    boot = np.median(x[rng.integers(0, len(x), (N_BOOT, len(x)))], axis=1)
    pairs = [float(np.median((a[i] - b[j]).dropna())) for i in range(2) for j in range(2)]
    return {"comparison": f"{first} - {second}", "songs": len(x), "median": float(np.median(x)),
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "first_better": int((x > 0).sum()), "seed_pair_min": min(pairs), "seed_pair_max": max(pairs)}


def summarize() -> None:
    sel = json.loads((OUT / "selection.json").read_text())
    chosen = sel["selected"]
    rows = [paired("C_dual", chosen), paired(chosen, "B_sym256")]
    if chosen != CANDIDATES[1e-3]:
        rows += [paired("C_dual", CANDIDATES[1e-3]), paired(chosen, CANDIDATES[1e-3])]
    pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False)

    def med(name):
        a = per_seed(name)
        return float(((a[0] + a[1]) / 2).median())

    lines = ["# Exploratory learning-rate check for the 5.8 ms wide short-window control", "",
             "Design: [`design.md`](design.md). Post hoc and exploratory; `C_dual` gets no extra learning-rate "
             "candidate, so the comparison is conservative toward the dual window.", "",
             "## Validation-only selection", "",
             "| setting | lr | best validation, seed 0 / seed 1 (dB) | mean |", "|---|---:|---|---:|"]
    for name, v in sel["candidates"].items():
        s = v["seeds"]
        lines.append(f"| {name} | {v['lr']:g} | {s['0']['best_val_si_sdr_median']:.3f} / "
                     f"{s['1']['best_val_si_sdr_median']:.3f} | {v['mean_best_val']:.3f} |")
    lines += ["", f"Selected: **{chosen}**.", "", "## Test songs (50, full length, mono; seeds averaged per song)", "",
              "| config | SI-SDR median (dB) |", "|---|---:|"]
    for name in ["B_sym256", chosen, "C_dual"]:
        lines.append(f"| {name} | {med(name):.2f} |")
    lines += ["", "| comparison | songs | median difference [95% CI] | first better | seed-pair medians |",
              "|---|---:|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['comparison']} | {r['songs']} | {r['median']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
                     f"| {r['first_better']} / {r['songs']} | {r['seed_pair_min']:+.2f} … {r['seed_pair_max']:+.2f} |")
    lines += ["", f"Bootstrap: {N_BOOT:,} song resamples, seed 0; intervals exclude training-seed variance. "
              "Per-song values: `results/eval/*_test.csv`; table: `paired.csv`."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    select() if args.select else summarize()
