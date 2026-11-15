"""Exploratory learning-rate check for the original short-window baselines B and E.

Design: results/baseline_lr_check/design.md.
  --select     pick each baseline's learning rate from validation logs only; writes selection.json and prints
               the newly selected config names that still need a test evaluation (one per line, may be empty)
  --summarize  paired test comparisons for the selected settings; writes summary.md and paired.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "baseline_lr_check"
EVAL = REPO / "results" / "eval"
PAIRS = {
    "5.8 ms": {"dual": "C_dual", "candidates": {1e-3: "B_sym256", 3e-4: "B_sym256_lr3e4"}},
    "2.9 ms": {"dual": "D_dual_2p9ms", "candidates": {1e-3: "E_sym128", 3e-4: "E_sym128_lr3e4"}},
}
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


def choose(table: dict, base: str, alt: str) -> str:
    """Validation-only rule: the alternative wins only if its mean is higher by more than the tie margin."""
    return alt if table[alt]["mean_best_val"] > table[base]["mean_best_val"] + TIE_DB else base


def select() -> list[str]:
    out, todo = {}, []
    for latency, spec in PAIRS.items():
        table = {}
        for lr, name in spec["candidates"].items():
            per_seed = {s: best_validation(REPO / "runs" / f"{name}_seed{s}") for s in SEEDS}
            mean = float(np.mean([v["best_val_si_sdr_median"] for v in per_seed.values()]))
            table[name] = {"lr": lr, "seeds": per_seed, "mean_best_val": mean}
            print(f"{latency} {name}: mean best validation {mean:.3f} dB", file=sys.stderr)
        base, alt = spec["candidates"][1e-3], spec["candidates"][3e-4]
        chosen = choose(table, base, alt)
        out[latency] = {"dual": spec["dual"], "candidates": table, "selected": chosen}
        if chosen != base:
            todo.append(chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(
        {"rule": "per baseline: mean over seeds of each seed's best validation median SI-SDR; "
                 "ties within 1e-9 dB keep 1e-3", "latencies": out}, indent=2))
    for name in todo:
        print(name)
    return todo


def per_seed(name: str) -> list[pd.Series]:
    return [pd.read_csv(EVAL / f"{name}_seed{s}_test.csv").set_index("track")["si_sdr"] for s in SEEDS]


def paired(first: str, second: str) -> dict:
    a, b = per_seed(first), per_seed(second)
    d = ((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2).dropna()
    x = d.to_numpy()
    rng = np.random.default_rng(0)
    draws = x[rng.integers(0, len(x), (N_BOOT, len(x)))]
    med, mean = np.median(draws, axis=1), draws.mean(axis=1)
    pairs = [float(np.median((a[i] - b[j]).dropna())) for i in range(2) for j in range(2)]
    return {"comparison": f"{first} - {second}", "songs": len(x), "median": float(np.median(x)),
            "ci_low": float(np.percentile(med, 2.5)), "ci_high": float(np.percentile(med, 97.5)),
            "mean": float(x.mean()), "mean_ci_low": float(np.percentile(mean, 2.5)),
            "mean_ci_high": float(np.percentile(mean, 97.5)),
            "first_better": int((x > 0).sum()), "seed_pair_min": min(pairs), "seed_pair_max": max(pairs)}


def summarize() -> None:
    sel = json.loads((OUT / "selection.json").read_text())["latencies"]
    rows, lines = [], [
        "# Exploratory learning-rate check for the original short-window baselines", "",
        "Design: [`design.md`](design.md). Post hoc and exploratory. Only the baselines get a second learning "
        "rate here, so the comparison is conservative toward the dual window.", "",
        "## Validation-only selection", "",
        "| latency | setting | lr | best validation, seed 0 / seed 1 (dB) | mean | selected |",
        "|---|---|---:|---|---:|:---:|"]
    for latency, v in sel.items():
        for name, c in v["candidates"].items():
            s = c["seeds"]
            lines.append(f"| {latency} | {name} | {c['lr']:g} | {s['0']['best_val_si_sdr_median']:.3f} / "
                         f"{s['1']['best_val_si_sdr_median']:.3f} | {c['mean_best_val']:.3f} | "
                         f"{'yes' if name == v['selected'] else ''} |")

    def med(name):
        a = per_seed(name)
        return float(((a[0] + a[1]) / 2).median())

    lines += ["", "## Test songs (50, full length, mono; seeds averaged per song)", "",
              "| latency | comparison | songs | median difference [95% CI] | mean difference [95% CI] | "
              "first better | seed-pair medians |", "|---|---|---:|---|---|---:|---|"]
    for latency, v in sel.items():
        base = next(n for n, c in v["candidates"].items() if c["lr"] == 1e-3)
        pairs = [paired(v["dual"], v["selected"])]
        if v["selected"] != base:
            pairs += [paired(v["dual"], base), paired(v["selected"], base)]
        for r in pairs:
            rows.append({"latency": latency, **r})
            lines.append(
                f"| {latency} | {r['comparison']} | {r['songs']} | {r['median']:+.2f} [{r['ci_low']:+.2f}, "
                f"{r['ci_high']:+.2f}] | {r['mean']:+.2f} [{r['mean_ci_low']:+.2f}, {r['mean_ci_high']:+.2f}] | "
                f"{r['first_better']} / {r['songs']} | {r['seed_pair_min']:+.2f} … {r['seed_pair_max']:+.2f} |")
    lines += ["", "| latency | config | SI-SDR median (dB) |", "|---|---|---:|"]
    for latency, v in sel.items():
        for name in dict.fromkeys([*(n for n, c in v["candidates"].items() if c["lr"] == 1e-3),
                                   v["selected"], v["dual"]]):
            lines.append(f"| {latency} | {name} | {med(name):.2f} |")
    lines += ["", f"Bootstrap: {N_BOOT:,} song resamples, seed 0; intervals exclude training-seed variance. "
              "Per-song values: `results/eval/*_test.csv`; table: `paired.csv`."]
    pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    select() if args.select else summarize()
