"""Frozen full-song failure diagnostics; original test metrics remain the primary evaluation.

Run all eight original B/C/D/E models, then score reference-defined active samples without
resetting recurrent state or removing any songs. Outputs are isolated in results/failure_analysis.
Use --allow-partial with a separate --out for smoke tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from dwmss.data import SR, STEMS, StemStore, list_tracks
from dwmss.failure_analysis import activity_mask, block_power, diagnose_signals, paired_bootstrap, power_db
from dwmss.model import MaskNet
from dwmss.plotstyle import AQUA, BLUE, GRAY, INK2, ORANGE, apply_style, plt
from dwmss.stft import DualWindowSTFT

CONFIGS = ["B_sym256", "C_dual", "E_sym128", "D_dual_2p9ms"]
RUNS = [f"{config}_seed{seed}" for config in CONFIGS for seed in (0, 1)]
PAIRS = [("C_dual", "B_sym256", "C − B at 5.8 ms"), ("D_dual_2p9ms", "E_sym128", "D − E at 2.9 ms")]
OUT = REPO / "results/failure_analysis"
ACTIVITY_CONDITIONS = [(30.0, -60.0), (40.0, -60.0), (50.0, -60.0), (40.0, -50.0), (40.0, -70.0)]
PROTOCOL = {"version": 1, "sample_rate": SR, "configs": CONFIGS, "seeds": [0, 1],
            "inference": "full continuous song; GRU state carried over 4096-frame chunks",
            "activity": {"block_samples": SR, "primary_relative_db": 40.0,
                         "sensitivity_relative_db": [30.0, 50.0], "floor_dbfs": -60.0,
                         "sensitivity_floor_dbfs": [-50.0, -70.0],
                         "conditions_relative_db_floor_dbfs": [list(x) for x in ACTIVITY_CONDITIONS],
                         "criterion": "reference vocal block RMS >= max(peak block RMS - relative_db, floor)",
                         "short_final_block": "included with actual sample count",
                         "scoring": "concatenate selected samples after full-song inference"},
            "decomposition": "mixture-predicted mask applied unchanged to vocal and combined accompaniment",
            "case_selection": "five lowest full-song paired differences plus positive-gain 25/50/75th ranked quantiles, from frozen saved eval",
            "full_song_metric_tolerance_db": 0.02, "bootstrap": {"draws": 10000, "seed": 0, "unit": "songs after averaging seed metrics"}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, frame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def cached(path, identity, run, track):
    try:
        data = json.loads(path.read_text())
        if data["identity"] != identity or data["payload_sha256"] != digest(data["payload"]):
            return None
        rows = data["payload"]["rows"]
        if len(rows) != len(ACTIVITY_CONDITIONS) or {(r["relative_db"], r["floor_dbfs"]) for r in rows} != set(ACTIVITY_CONDITIONS):
            return None
        if any(r["run"] != run or r["track"] != track for r in rows):
            return None
        for r in rows:
            if not np.isfinite([r[k] for k in ("whole_si_sdr", "whole_usdr", "active_fraction", "threshold_dbfs", "decomposition_relative_rms")]).all():
                return None
            if r["active_samples"] > 0 and not np.isfinite(r["active_si_sdr"]):
                return None
        return data["payload"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def saved_scores(run):
    path = REPO / f"results/eval/{run}_test.csv"
    frame = pd.read_csv(path)
    if len(frame) != 50 or frame.track.duplicated().any() or not np.isfinite(frame[["si_sdr", "usdr"]]).all().all():
        raise ValueError(f"incomplete original evaluation: {path}")
    return frame.set_index("track"), sha(path)


def make_cases():
    frames, hashes = [], {}
    for run in RUNS:
        frame, hashes[run] = saved_scores(run)
        part = frame.reset_index()[["track", "si_sdr"]]
        part["config"], part["seed"] = run.rsplit("_seed", 1)[0], int(run.rsplit("_seed", 1)[1])
        frames.append(part)
    per = pd.concat(frames).groupby(["track", "config"]).si_sdr.mean().unstack()
    cases = []
    for a, b, label in PAIRS:
        diffs = (per[a] - per[b]).sort_values()
        for rank, (track, value) in enumerate(diffs.head(5).items(), 1):
            cases.append({"pair": label, "first": a, "second": b, "track": track,
                          "selection": f"worst_{rank}", "saved_full_gain_db": float(value)})
        positive = diffs[diffs > 0]
        for q in (0.25, 0.5, 0.75):
            i = int(round(q * (len(positive) - 1)))
            track = positive.index[i]
            cases.append({"pair": label, "first": a, "second": b, "track": track,
                          "selection": f"positive_rank_quantile_{q:g}", "saved_full_gain_db": float(positive.iloc[i])})
    return {"selection": PROTOCOL["case_selection"], "saved_eval_sha256": hashes, "cases": cases}


@torch.no_grad()
def decompose(model, stft, vocal, accompaniment, device):
    mix = (vocal + accompaniment).astype(np.float32)
    x = torch.as_tensor(mix, device=device)[None]
    spectrum = stft.analysis(x)
    masks, state = [], None
    for start in range(0, spectrum.shape[1], 4096):
        mask, state = model(spectrum[:, start:start + 4096], stft.window_energy, state)
        masks.append(mask)
    mask = torch.cat(masks, dim=1)
    estimate = stft.synthesis(mask * spectrum, len(mix))[0].float().cpu().numpy()
    del spectrum, x, masks
    def component(signal):
        spectrum = stft.analysis(torch.as_tensor(signal, dtype=torch.float32, device=device)[None])
        return stft.synthesis(mask * spectrum, len(mix))[0].float().cpu().numpy()
    out_vocal = component(vocal)
    out_acc = component(accompaniment)
    error = np.asarray(estimate, np.float64) - out_vocal - out_acc
    relative = float(np.sqrt(np.mean(error ** 2) / (np.mean(np.asarray(estimate, np.float64) ** 2) + 1e-20)))
    absolute = float(np.max(np.abs(error)))
    if relative > 2e-5 or absolute > 2e-5:
        raise RuntimeError(f"failed mixture-mask linear decomposition: relative RMS {relative}, peak {absolute}")
    return estimate, out_vocal, out_acc, relative, absolute


def run_one(run, tracks, store, device, out, protocol_hash, code_hash, skip):
    directory = REPO / "runs" / run
    checkpoint = directory / "best.pt"
    ck_hash = sha(checkpoint)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ck["config"]
    if config["name"] != run.rsplit("_seed", 1)[0] or int(ck["seed"]) != int(run.rsplit("_seed", 1)[1]):
        raise ValueError(f"checkpoint identity mismatch: {run}")
    model = MaskNet(**config["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    stft = DualWindowSTFT(config["stft"]["n_analysis"], config["stft"]["n_synthesis"]).to(device)
    baseline, baseline_hash = saved_scores(run)
    eval_meta = json.loads((REPO / f"results/eval/{run}_test.json").read_text())
    if eval_meta["step"] != ck["step"]:
        raise ValueError(f"best checkpoint step differs from saved evaluation: {run}")
    run_info = {"checkpoint_sha256": ck_hash, "checkpoint_step": ck["step"], "saved_eval_sha256": baseline_hash,
                "config": config["name"], "seed": ck["seed"], "protocol_sha256": protocol_hash,
                "code_sha256": code_hash, "torch": torch.__version__, "device": str(device),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(out / "run_provenance" / f"{run}.json", run_info)
    rows, blocks = [], []
    for index, track in enumerate(tracks):
        sources = []
        for stem in STEMS:
            path = Path(store.dir) / track / f"{stem}.wav"
            stat = path.stat()
            sources.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        identity = {"checkpoint_sha256": ck_hash, "protocol_sha256": protocol_hash, "code_sha256": code_hash,
                    "saved_eval_sha256": baseline_hash, "sources": sources}
        cache_path = out / "cache" / run / (hashlib.sha256(track.encode()).hexdigest()[:20] + ".json")
        payload = cached(cache_path, identity, run, track) if skip else None
        reused = payload is not None
        if payload is None:
            n = store.lengths[track]
            vocal = store.read(track, "vocals", 0, n)
            acc = sum(store.read(track, stem, 0, n) for stem in STEMS[1:])
            estimate, out_vocal, out_acc, relative, peak = decompose(model, stft, vocal, acc, device)
            song_rows = []
            primary_activity = activity_mask(vocal, SR, 40.0, -60.0)[0]
            for threshold, floor in ACTIVITY_CONDITIONS:
                metrics = diagnose_signals(vocal, acc, estimate, out_vocal, out_acc, SR, threshold, floor)
                active = activity_mask(vocal, SR, threshold, floor)[0]
                metrics["same_activity_as_primary"] = int(np.array_equal(active, primary_activity))
                metrics["activity_mask_sha256"] = hashlib.sha256(np.packbits(active).tobytes()).hexdigest()
                for key, expected in (("whole_si_sdr", baseline.loc[track, "si_sdr"]), ("whole_usdr", baseline.loc[track, "usdr"])):
                    metrics[key + "_saved_difference_db"] = float(metrics[key] - expected)
                    if abs(metrics[key] - expected) > PROTOCOL["full_song_metric_tolerance_db"]:
                        raise RuntimeError(f"full-song metric mismatch {run}/{track}/{key}: {metrics[key]} vs {expected}")
                song_rows.append({"run": run, "config": config["name"], "seed": int(ck["seed"]), "track": track,
                                  "seconds": n / SR, "decomposition_relative_rms": relative,
                                  "decomposition_max_abs": peak, **metrics})
            block_data = {"run": run, "config": config["name"], "seed": int(ck["seed"]), "track": track,
                          "seconds": n / SR}
            for key, signal in (("vocal", vocal), ("accompaniment", acc), ("estimate", estimate),
                                ("preserved_vocal", out_vocal), ("residual_accompaniment", out_acc)):
                powers, counts = block_power(signal, SR)
                block_data[key + "_dbfs"] = power_db(powers).tolist()
            block_data["block_samples"] = counts.tolist()
            payload = {"rows": song_rows, "blocks": block_data}
            write_json(cache_path, {"identity": identity, "payload": payload, "payload_sha256": digest(payload)})
            del vocal, acc, estimate, out_vocal, out_acc
        rows.extend(payload["rows"])
        blocks.append(payload["blocks"])
        print(f"{run} [{index + 1}/{len(tracks)}] {track}: {'cache' if reused else 'computed'}", flush=True)
    frame = pd.DataFrame(rows)
    write_csv(out / f"{run}.csv", frame)
    write_json(out / "blocks" / f"{run}.json", blocks)
    write_json(out / "completions" / f"{run}.json", {**run_info, "tracks": tracks,
               "csv_sha256": sha(out / f"{run}.csv"), "blocks_sha256": sha(out / "blocks" / f"{run}.json")})


def paired_table(per):
    rows = []
    for first, second, label in PAIRS:
        if not {first, second} <= set(per.config):
            continue
        for relative, floor in ACTIVITY_CONDITIONS:
            current = per[(per.relative_db == relative) & (per.floor_dbfs == floor)]
            for metric in ("whole_si_sdr", "whole_usdr", "active_si_sdr", "active_usdr",
                           "active_vocal_loss_db", "active_accompaniment_attenuation_db"):
                if metric.startswith("whole") and (relative, floor) != (40.0, -60.0):
                    continue
                table = current.pivot(index="track", columns="config", values=metric)
                delta = (table[first] - table[second]).dropna()
                if len(delta):
                    rows.append({"pair": label, "first": first, "second": second, "metric": metric,
                                 "relative_db": relative, "floor_dbfs": floor, **paired_bootstrap(delta.to_numpy())})
    return pd.DataFrame(rows)


def seed_average(frame, partial=False):
    """Keep a metric undefined if either required seed has no defined measurement."""
    keys = ["config", "track", "relative_db", "floor_dbfs"]
    grouped = frame.groupby(keys)
    if not partial and not all(sorted(values) == [0, 1] for values in grouped.seed.agg(list)):
        raise ValueError("summary requires both distinct seeds for every song/config/condition")
    numeric = [column for column in frame.select_dtypes(include="number") if column not in {"seed", "relative_db", "floor_dbfs"}]
    return grouped[numeric].agg(lambda column: column.mean() if column.notna().all() else np.nan).reset_index()


def validate_frame(frame, complete, run, partial=False):
    expected = {(track, threshold, floor) for track in complete["tracks"] for threshold, floor in ACTIVITY_CONDITIONS}
    observed = set(frame[["track", "relative_db", "floor_dbfs"]].itertuples(index=False, name=None))
    if len(frame) != len(expected) or observed != expected or frame.duplicated(["track", "relative_db", "floor_dbfs"]).any():
        raise ValueError(f"missing or duplicate rows: {run}")
    config, seed = run.rsplit('_seed', 1)
    if set(frame.run) != {run} or set(frame.config) != {config} or set(frame.seed) != {int(seed)}:
        raise ValueError(f"incorrect run/config/seed identity: {run}")
    if not partial and len(complete["tracks"]) != 50:
        raise ValueError(f"incomplete run scope: {run}")
    required = ["whole_si_sdr", "whole_usdr", "active_samples", "active_fraction", "threshold_dbfs", "decomposition_relative_rms"]
    if not np.isfinite(frame[required].to_numpy()).all():
        raise ValueError(f"nonfinite required metrics: {run}")
    active = frame.active_samples > 0
    if not np.isfinite(frame.loc[active, ["active_si_sdr", "active_usdr"]].to_numpy()).all():
        raise ValueError(f"undefined score on nonempty active reference: {run}")


def summarize(out, partial=False):
    frames, complete_info = [], {}
    protocol = json.loads((out / "protocol.json").read_text())
    for run in RUNS:
        path = out / f"{run}.csv"
        if not path.exists():
            if partial:
                continue
            raise FileNotFoundError(path)
        complete = json.loads((out / "completions" / f"{run}.json").read_text())
        if complete["csv_sha256"] != sha(path) or complete["blocks_sha256"] != sha(out / "blocks" / f"{run}.json"):
            raise ValueError(f"result content changed after completion: {run}")
        if (complete["protocol_sha256"] != protocol["protocol_sha256"] or
                complete["code_sha256"] != protocol["code_sha256"]):
            raise ValueError(f"completion protocol/code identity mismatch: {run}")
        if sha(REPO / f"results/eval/{run}_test.csv") != complete["saved_eval_sha256"]:
            raise ValueError(f"saved original evaluation changed after inference: {run}")
        frame = pd.read_csv(path)
        validate_frame(frame, complete, run, partial)
        if sha(REPO / "runs" / run / "best.pt") != complete["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed after inference: {run}")
        frames.append(frame)
        complete_info[run] = complete
    full = pd.concat(frames, ignore_index=True)
    per = seed_average(full, partial)
    write_csv(out / "per_track.csv", per)
    paired = paired_table(per)
    write_csv(out / "paired.csv", paired)
    cases_data = json.loads((out / "cases.json").read_text())
    cases = []
    for case in cases_data["cases"]:
        selected = per[(per.track == case["track"]) & (per.relative_db == 40.0) & (per.floor_dbfs == -60.0)].set_index("config")
        if not {case["first"], case["second"]} <= set(selected.index):
            continue
        a, b = selected.loc[case["first"]], selected.loc[case["second"]]
        row = dict(case)
        row.update({"diagnostic_full_gain_db": a.whole_si_sdr - b.whole_si_sdr,
                    "active_gain_db": a.active_si_sdr - b.active_si_sdr,
                    "active_fraction": a.active_fraction, "active_reference_energy_fraction": a.active_reference_energy_fraction,
                    "threshold_dbfs": a.threshold_dbfs, "reference_vocal_dbfs": a.whole_reference_dbfs,
                    "vocal_to_accompaniment_db": a.vocal_to_accompaniment_db})
        for name, data in (("first", a), ("second", b)):
            for metric in ("whole_si_sdr", "active_si_sdr", "whole_usdr", "active_usdr", "whole_cosine", "active_cosine",
                           "active_vocal_loss_db", "active_accompaniment_attenuation_db", "inactive_output_dbfs",
                           "inactive_output_energy_fraction"):
                row[name + "_" + metric] = data[metric]
        cases.append(row)
    case_frame = pd.DataFrame(cases)
    write_csv(out / "cases.csv", case_frame)
    apply_style()
    if not paired.empty:
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.8), squeeze=False)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.87, bottom=0.10, hspace=0.65)
        for ax, (_, _, label) in zip(axes[:, 0], PAIRS):
            rows = paired[(paired.pair == label) & ((paired.metric == "whole_si_sdr") | (paired.metric == "active_si_sdr"))]
            labels = ["Full song" if row.metric == "whole_si_sdr" else f"Active: peak−{int(row.relative_db)}\nfloor {int(row.floor_dbfs)} dBFS" for row in rows.itertuples()]
            x = np.arange(len(rows))
            ax.errorbar(x, rows["median"], yerr=[rows["median"] - rows.median_lo, rows.median_hi - rows["median"]],
                        fmt="o", color=AQUA, capsize=4, markersize=6)
            ax.axhline(0, color=INK2, lw=1)
            ax.set_xticks(x, labels)
            ax.set_ylabel("Paired SI-SDR gain (dB)")
            ax.set_title(label)
            ax.grid(axis="y")
        fig.suptitle(("PARTIAL SMOKE: " if partial else "") + "Full-song and reference-active diagnostics", fontsize=15, fontweight="bold", x=0.03, ha="left")
        fig.text(0.03, 0.92, "Median paired gain ± song-bootstrap 95% CI; same full inference, two seeds averaged\n"
                 f"Fixed thresholds; {full.track.nunique()} songs; " + ("partial runs, not scientific results" if partial else "all 50 songs retained"), fontsize=10, color=INK2)
        fig.savefig(out / "fig_activity_sensitivity.png")
        plt.close(fig)
    for first, second, label in PAIRS:
        selected = case_frame[(case_frame.pair == label) & (case_frame.selection == "worst_1")] if not case_frame.empty else pd.DataFrame()
        if selected.empty:
            continue
        track = selected.iloc[0].track
        arrays = {}
        for config in (first, second):
            loaded = []
            for seed in (0, 1):
                file = out / "blocks" / f"{config}_seed{seed}.json"
                if file.exists():
                    loaded.extend(row for row in json.loads(file.read_text()) if row["track"] == track)
            if loaded:
                arrays[config] = {key: np.mean([r[key] for r in loaded], axis=0) for key in ("vocal_dbfs", "accompaniment_dbfs", "estimate_dbfs")}
        if len(arrays) != 2:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.85, bottom=0.11, hspace=0.40)
        t = np.arange(len(arrays[first]["vocal_dbfs"])) + .5
        axes[0].plot(t, arrays[first]["vocal_dbfs"], color=BLUE, label="Clean vocal")
        axes[0].plot(t, arrays[first]["accompaniment_dbfs"], color=GRAY, label="Accompaniment")
        axes[0].axhline(max(arrays[first]["vocal_dbfs"].max() - 40, -60), ls="--", color=INK2, label="Primary activity threshold")
        axes[0].set_title("Reference source levels")
        for config, color in ((first, AQUA), (second, ORANGE)):
            axes[1].plot(t, arrays[config]["estimate_dbfs"], color=color, label=config.split("_")[0])
        axes[1].set_title("Model output levels (seed-average block dBFS)")
        for ax in axes:
            ax.set_ylabel("1 s RMS (dBFS)")
            ax.set_ylim(-100, 0)
            ax.grid(axis="y")
            ax.legend(loc="upper right", fontsize=9)
        axes[1].set_xlabel("Time within full song (s)")
        fig.suptitle(("PARTIAL SMOKE: " if partial else "") + f"{track} · {label}", fontsize=15, fontweight="bold", x=.02, ha="left")
        fig.savefig(out / f"fig_worst_{first.split('_')[0]}_{second.split('_')[0]}.png")
        plt.close(fig)
    lines = ["# Frozen-model failure analysis", "", "Status: " + ("PARTIAL SMOKE ONLY." if partial else "complete."), "",
             "Original full-song, 50-track results remain the primary evaluation. This is a post-hoc diagnostic of the "
             "already inspected test results, not a new independent confirmatory test. No tracks are removed and no models "
             "are selected using these diagnostics.", "", "## Protocol", "",
             "Frozen B/C/D/E best checkpoints, two seeds. Each complete song is inferred continuously; recurrent state "
             "is carried across chunks. The same mask predicted from the mixture is separately applied to the vocal and "
             "combined accompaniment. Numerical decomposition and agreement with saved full-song SI-SDR/uSDR are checked.", "",
             "Activity uses only clean reference vocals in non-overlapping 1 s blocks, including the final short block "
             "with its actual sample count: RMS ≥ max(peak-block RMS − 40 dB, −60 dBFS). Fixed sensitivity thresholds "
             "are 30 and 50 dB below peak with the same floor, plus −50/−70 dBFS floors at relative 40 dB. "
             "Effective thresholds and identical-mask indicators are saved, because the absolute floor can dominate. "
             "Selected samples are concatenated only for scoring after "
             "continuous inference. Inactive intervals receive output-energy measurements, not SI-SDR. All thresholds "
             "are reported; none is chosen by test performance.", "",
             "## Paired full-song and active-sample results", "",
             "Seeds are averaged per song before paired song-bootstrap intervals (10,000 draws, seed 0). CIs do not "
             "include training-seed uncertainty. Positive vocal-loss differences mean more vocal energy retained; "
             "positive accompaniment-attenuation differences mean more accompaniment suppressed.", "",
             "| pair | metric | dB below peak / floor dBFS | songs | median difference [95% CI] | mean difference [95% CI] | positive songs |",
             "|---|---|---|---:|---|---|---:|"]
    for r in paired.itertuples():
        activity_label = '—' if r.metric.startswith('whole') else f'{int(r.relative_db)} / {int(r.floor_dbfs)}'
        lines.append(f"| {r.pair} | {r.metric} | {activity_label} | {r.songs} | "
                     f"{r.median:+.2f} [{r.median_lo:+.2f}, {r.median_hi:+.2f}] | "
                     f"{r.mean:+.2f} [{r.mean_lo:+.2f}, {r.mean_hi:+.2f}] | {r.better}/{r.songs} |")
    lines += ["", "## Selected cases", "", "Case selection was fixed from saved full-song differences before computing "
              "activity diagnostics: five lowest gains and three positive-gain rank quantiles for each latency pair. "
              "This deliberately includes failures; these cases are not representative estimates of all-song quality.", "",
              "| pair | selection | track | full gain | active gain (primary) | active samples / vocal energy kept | threshold | vocal RMS | vocal/acc energy |",
              "|---|---|---|---:|---:|---|---:|---:|---:|"]
    for r in case_frame.itertuples():
        lines.append(f"| {r.pair} | {r.selection} | {r.track} | {r.saved_full_gain_db:+.2f} | {r.active_gain_db:+.2f} | "
                     f"{100 * r.active_fraction:.1f}% / {100 * r.active_reference_energy_fraction:.1f}% | "
                     f"{r.threshold_dbfs:.1f} dBFS | {r.reference_vocal_dbfs:.1f} dBFS | {r.vocal_to_accompaniment_db:.1f} dB |")
    happy = per[(per.track == "PR - Happy Daze") & (per.relative_db == 40) & (per.floor_dbfs == -60)].set_index("config")
    if {"D_dual_2p9ms", "E_sym128"} <= set(happy.index):
        lines += ["", "## Extreme-case details: PR - Happy Daze", "",
                  "| model | full SI-SDR | active SI-SDR | full uSDR | active uSDR | full cosine | active cosine | active vocal loss | active acc attenuation | inactive output | inactive output-energy share |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for config, r in happy.iterrows():
            lines.append(f"| {config} | {r.whole_si_sdr:.2f} | {r.active_si_sdr:.2f} | {r.whole_usdr:.2f} | {r.active_usdr:.2f} | "
                         f"{r.whole_cosine:.5f} | {r.active_cosine:.5f} | {r.active_vocal_loss_db:+.2f} dB | "
                         f"{r.active_accompaniment_attenuation_db:.2f} dB | {r.inactive_output_dbfs:.1f} dBFS | {100 * r.inactive_output_energy_fraction:.1f}% |")
        lines += ["", "Activity sensitivity for this track (reference-only, identical across models):", "",
                  "| relative dB | floor dBFS | effective threshold dBFS | active samples | reference energy kept | same mask as primary |",
                  "|---:|---:|---:|---:|---:|---|"]
        for r in per[(per.track == 'PR - Happy Daze') & (per.config == 'D_dual_2p9ms')].itertuples():
            lines.append(f"| {r.relative_db:.0f} | {r.floor_dbfs:.0f} | {r.threshold_dbfs:.1f} | "
                         f"{100 * r.active_fraction:.1f}% | {100 * r.active_reference_energy_fraction:.2f}% | "
                         f"{'yes' if r.same_activity_as_primary == 1 else 'no'} |")
    lines += ["", "## Interpretation limits", "",
              "A low-energy or sparse reference can make full-song projection metrics sensitive to residual output, "
              "but it is not evidence that a bad score is merely an artifact. Check active SI-SDR, uSDR, the uncentered "
              "cosine/projection coefficient, vocal preservation and residual accompaniment together. A score change "
              "after masking activity changes the evaluation question and does not replace the full-song result. "
              "Preserved-vocal energy alone is not perceptual naturalness; no listener study was performed.", "",
              "`per_track.csv` contains seed-averaged metrics; the eight run CSVs retain individual seeds. `cases.csv` "
              "contains case-level details. Block-level numeric curves are in `blocks/`; no MUSDB audio is exported. "
              "Source files are identified by path/size/mtime, not content hashes. Checkpoints, code and output CSVs "
              "have SHA-256 provenance, and per-song caches validate identities and payload hashes."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(out / "summary_provenance.json", {"runs": complete_info, "partial": partial,
               "outputs": {p.name: sha(p) for p in out.glob('*') if p.is_file() and p.suffix in {'.csv', '.png', '.md'}}})
    print(out / "summary.md", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--songs", nargs="+")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--musdb-root", default="data/musdb18hq")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if (args.songs or args.runs != RUNS) and not args.allow_partial:
        ap.error("partial runs/songs require --allow-partial and an isolated --out")
    if args.allow_partial and args.out.resolve() == OUT.resolve():
        ap.error("partial smoke outputs must use an isolated --out")
    if not set(args.runs) <= set(RUNS) or len(set(args.runs)) != len(args.runs):
        ap.error("runs must be unique original B/C/D/E seed0/1 run names")
    args.out.mkdir(parents=True, exist_ok=True)
    code = {str(p.relative_to(REPO)): sha(p) for p in [Path(__file__), REPO / "dwmss/failure_analysis.py",
            REPO / "dwmss/model.py", REPO / "dwmss/stft.py", REPO / "dwmss/data.py", REPO / "dwmss/metrics.py"]}
    protocol_hash, code_hash = digest(PROTOCOL), digest(code)
    protocol = {"protocol": PROTOCOL, "protocol_sha256": protocol_hash, "code": code, "code_sha256": code_hash}
    prior = args.out / "protocol.json"
    if prior.exists() and json.loads(prior.read_text()) != protocol:
        raise RuntimeError("output protocol/code changed; use a new output directory")
    write_json(prior, protocol)
    cases = make_cases()
    if (args.out / "cases.json").exists() and json.loads((args.out / "cases.json").read_text()) != cases:
        raise RuntimeError("saved case-selection inputs changed")
    write_json(args.out / "cases.json", cases)
    if not args.summary_only:
        tracks = list_tracks(args.musdb_root, "test")
        if len(tracks) != 50:
            raise ValueError(f"expected exactly 50 test tracks, found {len(tracks)}")
        if args.songs:
            if len(set(args.songs)) != len(args.songs) or not set(args.songs) <= set(tracks):
                ap.error("songs must be unique test-track names")
            tracks = sorted(args.songs)
        store = StemStore(args.musdb_root, "test", tracks)
        device = torch.device(args.device)
        for run in args.runs:
            run_one(run, tracks, store, device, args.out, protocol_hash, code_hash, args.skip_existing)
    summarize(args.out, args.allow_partial)


if __name__ == "__main__":
    main()
