"""Freeze short-window controls matched to C/D's model MACs, without reading quality results.

Only the GRU width (and its input/output projections) changes. The grid, objective and
tolerance are fixed below; CPU time is measured afterwards, never used for selection.
Usage: python -I scripts/select_compute_matched.py [--verify-torch] [--check]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dwmss.config import load_config  # noqa: E402

WIDTHS = tuple(range(256, 1025, 8))
TOLERANCE = 0.01
PAIRS = (("B_sym256", "C_dual", "B_sym256_compute"),
         ("E_sym128", "D_dual_2p9ms", "E_sym128_compute"))
SEEDS = (0, 1)


def analytic_description(cfg: dict) -> dict:
    """Layer-shape counts matching dwmss.complexity, without constructing any network."""
    m = cfg["model"]
    c1, c2, bands = m.get("c1", 16), m.get("c2", 32), m.get("n_bands", 64)
    h, hh, pos = m.get("hidden", 256), m.get("head_hidden", 32), m.get("n_pos", 8)
    n, synth = cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"]
    bins, hop = n // 2 + 1, synth // 2
    l2 = (bins + 1) // 2
    l3 = (l2 + 1) // 2
    macs = (bins * c1 * 9 + l2 * c2 * c1 * 7 + l3 * c2 * c2 * 7
            + l3 * bands * c2 + c2 * bands * h + 6 * h * h
            + h * hh + bins * c1 * hh + bins * hh)
    params = (c1 * 9 + c1 + c2 * c1 * 7 + c2 + c2 * c2 * 7 + c2
              + c2 * bands * h + h + 6 * h * h + 6 * h
              + h * hh + hh + c1 * hh + pos * hh + hh + 1)
    return {"hidden": h, "params": params, "n_bins": bins, "hop_samples": hop,
            "algorithmic_latency_ms": 1000 * synth / 44100,
            "macs_per_frame": macs, "macs_per_frame_M": macs / 1e6,
            "gmacs_per_s": macs * 44100 / hop / 1e9}


def choose_width(base: dict, target_macs: int) -> tuple[dict, list[dict]]:
    candidates = []
    for width in WIDTHS:
        cfg = {**base, "model": {**base["model"], "hidden": width}}
        row = analytic_description(cfg)
        row["relative_mac_error"] = row["macs_per_frame"] / target_macs - 1
        candidates.append(row)
    selected = min(candidates, key=lambda r: (abs(r["relative_mac_error"]), r["hidden"]))
    if abs(selected["relative_mac_error"]) > TOLERANCE:
        raise ValueError("The fixed width grid has no control within the 1% MAC tolerance")
    return selected, candidates


def make_design(repo: Path = REPO) -> dict:
    pairs = []
    source_hashes = {}
    for filename in ("base.yaml", "B_sym256.yaml", "C_dual.yaml", "E_sym128.yaml", "D_dual_2p9ms.yaml"):
        source_hashes[filename] = hashlib.sha256((repo / "configs" / filename).read_bytes()).hexdigest()
    for baseline, reference, control in PAIRS:
        base = load_config(str(repo / "configs" / f"{baseline}.yaml"))
        ref = load_config(str(repo / "configs" / f"{reference}.yaml"))
        target = analytic_description(ref)
        selected, candidates = choose_width(base, target["macs_per_frame"])
        actual = load_config(str(repo / "configs" / f"{control}.yaml"))
        expected = {**base, "name": control, "model": {**base["model"], "hidden": selected["hidden"]}}
        if actual != expected:
            raise ValueError(f"{control}: must differ from {baseline} only in name and selected hidden width")
        if actual["train"]["max_steps"] != 200000:
            raise ValueError("The prespecified training budget is 200000 updates")
        if base["stft"]["n_synthesis"] != ref["stft"]["n_synthesis"]:
            raise ValueError("MAC-matched comparisons must also have the same hop and latency")
        pairs.append({"baseline": baseline, "reference": reference, "control": control,
                      "reference_description": target, "baseline_description": analytic_description(base),
                      "selected": selected, "candidates": candidates,
                      "resolved_control_config": actual})
    return {"schema_version": 1, "selection_stage": "fixed before training the new controls",
            "objective": "smallest absolute relative model MACs/frame error; lower width breaks ties",
            "width_grid": list(WIDTHS), "relative_mac_tolerance": TOLERANCE,
            "matched_resource": "model MACs per STFT frame (and per second within each same-hop pair)",
            "not_matched": ["parameter count", "measured CPU time", "STFT/iSTFT cost", "training wall-clock time"],
            "quality_results_used_for_selection": False,
            "training": {"steps": 200000, "seeds": list(SEEDS), "checkpoint": "best validation median SI-SDR",
                         "other_settings": "unchanged from configs/base.yaml; equal crop, batch, loss and augmentation"},
            "evaluation": {"songs": 50, "full_length": True, "mono": True,
                           "primary_statistic": "median paired per-song SI-SDR difference after seed averaging",
                           "bootstrap": "10000 song resamples; training-seed variation reported separately",
                           "cpu_profile": "post-hoc full streaming pipeline, trained seed0 checkpoints, same job and CPU"},
            "interpretation": "resource-allocation comparison: a larger short-window GRU versus the original dual window",
            "source_config_sha256": source_hashes, "pairs": pairs}


def markdown(design: dict) -> str:
    lines = ["# Prespecified MAC-matched short-window controls", "",
             "Widths are chosen from the fixed 256:8:1024 grid by model MACs before the new controls are trained. "
             "No quality results or CPU timings enter selection. Complete candidate counts are in `candidates.csv` and `design.json`.", "",
             "| control | reference | GRU width | parameters | model MMAC/frame | target MMAC/frame | MAC error |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for p in design["pairs"]:
        s, ref = p["selected"], p["reference_description"]
        lines.append(f"| {p['control']} | {p['reference']} | {s['hidden']} | {s['params']:,} | "
                     f"{s['macs_per_frame_M']:.6f} | {ref['macs_per_frame_M']:.6f} | {s['relative_mac_error']:+.4%} |")
    lines += ["", "Only `model.hidden` changes relative to each original symmetric-window baseline. "
              "This also widens the GRU input projection and state-to-mask projection; all other layer settings are fixed.", "",
              "Both controls use 200,000 updates, seeds 0 and 1, the original data split/crops/batches/augmentation/loss, "
              "and the checkpoint with best validation median SI-SDR. Test quality never selects a width or checkpoint.", "",
              "The primary comparisons are C minus the larger B at 5.8 ms and D minus the larger E at 2.9 ms. "
              "Also report the wider controls versus their original width-256 baselines, paired over all 50 full-length "
              "mono test songs after averaging seeds. Confidence intervals resample songs; they do not include training-seed variance.", "",
              "This matches model MACs to within 1%, not parameter count or measured CPU time. "
              "The larger short-window models have more parameters; their training wall-clock time can differ. "
              "The STFT/iSTFT operation count is outside the matching budget. CPU runtime is an outcome measured "
              "afterwards for all six models in one ONNX Runtime job on one CPU thread, with trained seed-0 checkpoints. "
              "No claim of CPU-matched or device end-to-end latency follows from this design.", "",
              "Run `python -I scripts/select_compute_matched.py --check` before training and reporting. "
              "The selector refuses to replace a frozen design with a different one."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "compute_matched")
    ap.add_argument("--check", action="store_true", help="verify the frozen design without writing")
    ap.add_argument("--verify-torch", action="store_true", help="cross-check selected counts with the actual model")
    args = ap.parse_args()
    design = make_design()
    if args.verify_torch:
        from dwmss.complexity import describe
        for p in design["pairs"]:
            for name in (p["baseline"], p["reference"], p["control"]):
                cfg = load_config(str(REPO / "configs" / f"{name}.yaml"))
                analytic, actual = analytic_description(cfg), describe(cfg)
                if analytic["params"] != actual["params"] or abs(analytic["macs_per_frame_M"] - actual["macs_per_frame_M"]) > 1e-10:
                    raise ValueError(f"Analytic counts disagree with the actual model: {name}")
    path = args.out / "design.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError(f"Frozen design differs from current inputs: {path}; do not change the budget after training")
    if args.check:
        if not path.exists():
            raise FileNotFoundError(f"Freeze the design first: {path}")
        print(f"verified frozen MAC-matched design: {path}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(design, indent=2) + "\n")
    (args.out / "design.md").write_text(markdown(design))
    rows = [{"control": p["control"], "reference": p["reference"], **r,
             "selected": r["hidden"] == p["selected"]["hidden"]}
            for p in design["pairs"] for r in p["candidates"]]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (args.out / "candidates.csv").write_text(buf.getvalue())
    print(markdown(design))


if __name__ == "__main__":
    main()
