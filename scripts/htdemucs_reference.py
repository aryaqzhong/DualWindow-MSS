"""Offline reference: pretrained HTDemucs on the same 50 test songs, input and metric as the causal models.

HTDemucs (Rouard, Massa & Défossez, ICASSP 2023) is non-causal and processes ~7.8 s segments, so it has no
low-latency operating point; it is reported only as a separately labeled offline reference, not as a point on
the latency axis. Its training data (MUSDB18-HQ train + 800 extra songs) differs from ours.

Input: the same mono mixture as in scripts/evaluate.py (vocals + accompaniment stems, channel average),
duplicated to two channels because HTDemucs is a stereo model; the estimated vocals are averaged back to mono.
The mixture is normalized as in demucs' own `separate` command. Metric: SI-SDR over the full song.

Requires demucs 4.0.1 (`pip install demucs==4.0.1`); the pretrained weights are downloaded on first use.
`--extra-path` adds a directory that holds demucs to the import path (`python -I` ignores PYTHONPATH).
Usage: python -I scripts/htdemucs_reference.py [--device cuda] [--max-tracks K] [--extra-path DIR]
Outputs: results/eval/htdemucs_test.csv (per track), results/eval/htdemucs_test.json (summary)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dwmss.data import STEMS, StemStore, list_tracks  # noqa: E402
from dwmss.metrics import si_sdr, usdr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-tracks", type=int)
    ap.add_argument("--musdb-root", default="data/musdb18hq")
    ap.add_argument("--extra-path", help="directory containing the demucs package")
    args = ap.parse_args()
    if args.extra_path:
        sys.path.insert(0, args.extra_path)
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    device = torch.device(args.device)
    model = get_model("htdemucs")
    model.eval().to(device)
    vi = model.sources.index("vocals")
    n_params = sum(p.numel() for p in model.parameters())
    tracks = list_tracks(args.musdb_root, "test")
    tracks = tracks[:args.max_tracks] if args.max_tracks else tracks
    store = StemStore(args.musdb_root, "test", tracks)
    rows = []
    for i, t in enumerate(tracks):
        n = store.lengths[t]
        voc = store.read(t, "vocals", 0, n)
        acc = sum(store.read(t, s, 0, n) for s in STEMS[1:])
        mix = (voc + acc).astype(np.float32)
        wav = torch.from_numpy(np.stack([mix, mix]))
        ref = wav.mean(0)
        mu, sd = ref.mean(), ref.std()
        t0 = time.time()
        with torch.no_grad():
            out = apply_model(model, ((wav - mu) / sd)[None].to(device), device=device, shifts=1, split=True,
                              overlap=0.25, progress=False)[0]
        est = (out[vi] * sd.to(out.device) + mu.to(out.device)).mean(0).cpu().numpy()
        row = {"track": t, "seconds": n / 44100, "si_sdr": si_sdr(est, voc), "usdr": usdr(est, voc),
               "si_sdr_mixture": si_sdr(mix, voc), "processing_s": time.time() - t0}
        row["si_sdri"] = row["si_sdr"] - row["si_sdr_mixture"]
        rows.append(row)
        print(f"[{i + 1}/{len(tracks)}] {t}: SI-SDR {row['si_sdr']:.2f} dB ({row['processing_s']:.1f} s)", flush=True)
    df = pd.DataFrame(rows)
    out_dir = REPO / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "htdemucs_test.csv", index=False)
    summ = {"model": "htdemucs (demucs 4.0.1)", "params": int(n_params), "input": "mono mixture duplicated to stereo",
            "device": str(device), "tracks": len(df)}
    for c in ("si_sdr", "si_sdri", "usdr"):
        summ[f"{c}_median"], summ[f"{c}_mean"] = float(df[c].median()), float(df[c].mean())
    (out_dir / "htdemucs_test.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
