"""CPU streaming benchmark of the full per-hop pipeline (STFT + model + mask + iSTFT), one thread.

Timing does not depend on the weights, so configs without a trained checkpoint are benchmarked with
random weights (reported in the output). Run the final numbers on a compute node, not a login node.

Usage: python -I scripts/benchmark_streaming.py --configs configs/A_sym1024.yaml configs/B_sym256.yaml \
           configs/C_dual.yaml [--runs-dir runs --seed 0] [--seconds 30] [--backends torch onnx]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402

from dwmss.complexity import macs_per_frame  # noqa: E402
from dwmss.config import load_config  # noqa: E402
from dwmss.export import OnnxStep, export_onnx  # noqa: E402
from dwmss.model import MaskNet, count_parameters  # noqa: E402
from dwmss.stft import DualWindowSTFT  # noqa: E402
from dwmss.streaming import StreamingSeparator  # noqa: E402

SR = 44100


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def time_hops(fn, hops, warmup):
    for x in hops[:warmup]:
        fn(x)
    out = np.empty(len(hops) - warmup)
    for i, x in enumerate(hops[warmup:]):
        t = time.perf_counter_ns()
        fn(x)
        out[i] = (time.perf_counter_ns() - t) / 1e3
    return out  # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--backends", nargs="+", default=["torch", "onnx"])
    ap.add_argument("--out", default=str(REPO / "results" / "latency_benchmark"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        stft = DualWindowSTFT(cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"])
        model = MaskNet(**cfg["model"]).eval()
        ckpt = Path(args.runs_dir) / f"{cfg['name']}_seed{args.seed}" / "best.pt"
        weights = "random"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
            weights = str(ckpt.relative_to(REPO)) if ckpt.is_relative_to(REPO) else str(ckpt)
        R, n_hops = stft.R, int(args.seconds * SR / stft.R) + args.warmup
        audio = (rng.standard_normal(n_hops * R) * 0.1).astype(np.float32)
        hops = [audio[i * R:(i + 1) * R] for i in range(n_hops)]
        hop_us = 1e6 * R / SR
        # DSP-only cost (window, rFFT, irFFT, overlap-add) with a unit mask.
        st_dsp = StreamingSeparator(None, stft)
        st_dsp._mask = lambda X: np.ones(X.shape, dtype=np.float32)
        dsp = time_hops(st_dsp.process, hops, args.warmup)
        macs = macs_per_frame(model, stft.n_bins)
        for backend in args.backends:
            if backend == "torch":
                st = StreamingSeparator(model, stft)
            else:
                path = os.path.join(tempfile.mkdtemp(), f"{cfg['name']}.onnx")
                export_onnx(model, stft.n_bins, path)
                st = StreamingSeparator(None, stft, step=OnnxStep(path, model.gru.hidden_size, threads=1))
            t = time_hops(st.process, hops, args.warmup)
            rows.append({
                "config": cfg["name"], "backend": backend, "weights": weights, "N": stft.N, "M": stft.M,
                "hop_samples": R, "hop_ms": hop_us / 1e3, "algorithmic_latency_ms": 1e3 * stft.M / SR,
                "frames_per_s": SR / R, "params": count_parameters(model),
                "macs_per_frame_M": macs["total"] / 1e6, "gmacs_per_s": macs["total"] * SR / R / 1e9,
                "p50_us": float(np.percentile(t, 50)), "p95_us": float(np.percentile(t, 95)),
                "p99_us": float(np.percentile(t, 99)), "mean_us": float(t.mean()),
                "rtf": float(t.mean() / hop_us), "deadline_miss_rate": float(np.mean(t > hop_us)),
                "dsp_mean_us": float(dsp.mean()), "hops_timed": len(t),
            })
            r = rows[-1]
            print(f"{r['config']:14s} {backend:5s} hop={r['hop_ms']:.2f}ms p50={r['p50_us']:.0f}us p95={r['p95_us']:.0f}us "
                  f"p99={r['p99_us']:.0f}us RTF={r['rtf']:.3f} miss={r['deadline_miss_rate']:.2%} "
                  f"(DSP {r['dsp_mean_us']:.0f}us, {r['gmacs_per_s']:.2f} GMAC/s)", flush=True)
    env = {"cpu": cpu_name(), "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "threads": 1, "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
           "numpy": np.__version__, "python": platform.python_version(), "seconds_timed": args.seconds,
           "warmup_hops": args.warmup, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".json").write_text(json.dumps({"env": env, "rows": rows}, indent=1))
    L = ["# CPU streaming benchmark (full pipeline per hop, 1 thread)", "",
         "Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()), "",
         "| config | backend | weights | algorithmic latency (ms) | hop (ms) | P50 / P95 / P99 (µs) | RTF | "
         "deadline misses | DSP only (µs) | params | GMAC/s |", "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['config']} | {r['backend']} | {r['weights']} | {r['algorithmic_latency_ms']:.1f} | {r['hop_ms']:.2f} | "
                 f"{r['p50_us']:.0f} / {r['p95_us']:.0f} / {r['p99_us']:.0f} | {r['rtf']:.3f} | "
                 f"{r['deadline_miss_rate']:.2%} | {r['dsp_mean_us']:.0f} | {r['params']:,} | {r['gmacs_per_s']:.2f} |")
    L += ["", "RTF = mean processing time per hop / hop duration. Algorithmic latency (synthesis window) is separate "
              "from processing time; end-to-end latency with an audio device is not measured here."]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
