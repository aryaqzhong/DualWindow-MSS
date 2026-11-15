"""Train one window configuration. Resumes from <out>/last.pt if it exists.

Usage: python -I scripts/train.py --config configs/C_dual.yaml --seed 0 --out runs/C_dual_seed0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dwmss.config import load_config  # noqa: E402
from dwmss.data import RandomMixStream, StemStore, load_tracks_mono, train_valid_split  # noqa: E402
from dwmss.losses import WaveMRSTFTLoss  # noqa: E402
from dwmss.metrics import si_sdr  # noqa: E402
from dwmss.model import MaskNet, count_parameters  # noqa: E402
from dwmss.separator import separate  # noqa: E402
from dwmss.stft import DualWindowSTFT  # noqa: E402


def lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return 0.01 + 0.99 * step / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def validate(model, stft, tracks, baseline):
    model.eval()
    scores = {}
    for name, mix, voc in tracks:
        scores[name] = si_sdr(separate(model, stft, mix), voc)
    model.train()
    vals = np.array(list(scores.values()))
    imp = np.array([scores[n] - baseline[n] for n in scores])
    return {"si_sdr_median": float(np.median(vals)), "si_sdr_mean": float(vals.mean()),
            "si_sdri_median": float(np.median(imp)), "per_track": scores}


def save(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device")
    for k in ("max_steps", "batch_size", "num_workers", "val_every", "log_every"):
        ap.add_argument("--" + k.replace("_", "-"), type=int)
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--val-tracks", type=int, help="use only the first k validation tracks (smoke tests)")
    ap.add_argument("--val-seconds", type=float, help="truncate validation tracks (smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for k in ("max_steps", "batch_size", "num_workers", "val_every", "log_every", "max_hours"):
        if getattr(args, k) is not None:
            cfg["train"][k] = getattr(args, k)
    tc, dc = cfg["train"], cfg["data"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_tracks, valid_tracks = train_valid_split(dc["musdb_root"])
    if args.val_tracks:
        valid_tracks = valid_tracks[:args.val_tracks]
    stream = RandomMixStream(StemStore(dc["musdb_root"], "train", train_tracks), seconds=dc["seconds"],
                             seed=args.seed, p_remix=dc["p_remix"], gain_db=dc["gain_db"],
                             p_bleed=dc["p_bleed"], bleed_db=dc["bleed_db"])
    nw = int(tc["num_workers"])
    loader = torch.utils.data.DataLoader(stream, batch_size=int(tc["batch_size"]), num_workers=nw,
                                         pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                                         prefetch_factor=4 if nw > 0 else None)
    valid = load_tracks_mono(dc["musdb_root"], "train", valid_tracks, args.val_seconds)
    baseline = {name: si_sdr(mix, voc) for name, mix, voc in valid}

    stft = DualWindowSTFT(cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"]).to(device)
    model = MaskNet(**cfg["model"]).to(device)
    loss_fn = WaveMRSTFTLoss(**cfg["loss"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(int(tc["warmup_steps"]), int(tc["max_steps"]), float(tc["min_lr_ratio"])))

    step, best = 0, -float("inf")
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        print(f"resumed from step {step} (best val SI-SDR {best:.2f} dB)", flush=True)
    meta = {"config": cfg, "seed": args.seed, "params": count_parameters(model), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
            "latency_ms": 1000 * stft.latency_samples / 44100, "hop": stft.hop, "n_bins": stft.n_bins,
            "valid_tracks": valid_tracks, "baseline_si_sdr": baseline}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{cfg['name']}: N={stft.N} M={stft.M} hop={stft.hop} bins={stft.n_bins} "
          f"params={meta['params']} device={device}", flush=True)

    logf = open(out / "log.jsonl", "a")
    t_start, t_last, n_last = time.time(), time.time(), step
    max_steps, deadline = int(tc["max_steps"]), t_start + 3600 * float(tc["max_hours"])
    it = iter(loader)
    model.train()
    while step < max_steps:
        mix, voc = next(it)
        mix, voc = mix.to(device, non_blocking=True), voc.to(device, non_blocking=True)
        X = stft.analysis(mix)
        mask, _ = model(X, stft.window_energy)
        est = stft.synthesis(mask * X, mix.shape[-1])
        loss, parts = loss_fn(est, voc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"])))
        opt.step()
        sched.step()
        step += 1
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step % int(tc["log_every"]) == 0:
            now = time.time()
            rec = {"step": step, "loss": loss_value, **parts, "grad_norm": gnorm,
                   "lr": sched.get_last_lr()[0], "steps_per_s": (step - n_last) / (now - t_last),
                   "elapsed_h": (now - t_start) / 3600}
            if device.type == "cuda":
                rec["max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
            t_last, n_last = now, step
        out_of_time = time.time() > deadline
        if step % int(tc["val_every"]) == 0 or step == max_steps or out_of_time:
            v = validate(model, stft, valid, baseline)
            improved = v["si_sdr_median"] > best
            best = max(best, v["si_sdr_median"])
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "step": step, "best": best, "config": cfg, "seed": args.seed}
            save(out / "last.pt", state)
            if improved:
                save(out / "best.pt", state)
            rec = {"step": step, "val_si_sdr_median": v["si_sdr_median"], "val_si_sdr_mean": v["si_sdr_mean"],
                   "val_si_sdri_median": v["si_sdri_median"], "best": best, "improved": improved}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
            if out_of_time:
                print("stopping: time limit reached (resubmit to continue)", flush=True)
                break
    if step >= max_steps:
        (out / "DONE").write_text(f"finished {step} steps\n")
    logf.close()


if __name__ == "__main__":
    main()
