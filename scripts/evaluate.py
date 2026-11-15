"""Evaluate a trained run on MUSDB18-HQ test songs (full length, mono), optionally with the oracle reference.

Usage: python -I scripts/evaluate.py --run runs/C_dual_seed0 [--ckpt best.pt] [--subset test]
                                     [--max-tracks K] [--oracle] [--device cuda]
Outputs: results/eval/<run>_<subset>.csv (per track) and results/eval/<run>_<subset>.json (summary).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dwmss.data import STEMS, StemStore, list_tracks  # noqa: E402
from dwmss.metrics import si_sdr, usdr  # noqa: E402
from dwmss.model import MaskNet  # noqa: E402
from dwmss.separator import separate  # noqa: E402
from dwmss.stft import DualWindowSTFT  # noqa: E402


@torch.no_grad()
def oracle_irm2(stft, voc, acc, device):
    v = torch.tensor(voc, device=device)[None]
    a = torch.tensor(acc, device=device)[None]
    V, A = stft.analysis(v), stft.analysis(a)
    pv, pa = V.real ** 2 + V.imag ** 2, A.real ** 2 + A.imag ** 2
    est = stft.synthesis(pv / (pv + pa + 1e-8) * (V + A), v.shape[-1])
    return est[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--subset", choices=["test", "valid"], default="test")
    ap.add_argument("--max-tracks", type=int)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run = Path(args.run)
    ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    stft = DualWindowSTFT(cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"]).to(device)
    model = MaskNet(**cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    root = cfg["data"]["musdb_root"]
    if args.subset == "test":
        folder, tracks = "test", list_tracks(root, "test")
    else:
        from dwmss.data import train_valid_split
        folder, tracks = "train", train_valid_split(root)[1]
    tracks = tracks[:args.max_tracks] if args.max_tracks else tracks
    store = StemStore(root, folder, tracks)
    rows = []
    for i, t in enumerate(tracks):
        n = store.lengths[t]
        voc = store.read(t, "vocals", 0, n)
        acc = sum(store.read(t, s, 0, n) for s in STEMS[1:])
        mix = (voc + acc).astype(np.float32)
        est = separate(model, stft, mix)
        row = {"track": t, "seconds": n / 44100, "si_sdr": si_sdr(est, voc), "usdr": usdr(est, voc),
               "si_sdr_mixture": si_sdr(mix, voc)}
        row["si_sdri"] = row["si_sdr"] - row["si_sdr_mixture"]
        if args.oracle:
            o = oracle_irm2(stft, voc, acc, device)
            row["oracle_irm2_si_sdr"], row["oracle_irm2_usdr"] = si_sdr(o, voc), usdr(o, voc)
        rows.append(row)
        print(f"[{i + 1}/{len(tracks)}] {t}: SI-SDR {row['si_sdr']:.2f} dB (SI-SDRi {row['si_sdri']:+.2f})", flush=True)
    df = pd.DataFrame(rows)
    out = REPO / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{run.name}_{args.subset}"
    df.to_csv(out / f"{stem}.csv", index=False)
    summ = {"run": str(run), "ckpt": args.ckpt, "step": ck.get("step"), "config": cfg["name"],
            "n_tracks": len(df), "subset": args.subset}
    for c in [c for c in df.columns if c not in ("track", "seconds")]:
        summ[f"{c}_median"], summ[f"{c}_mean"] = float(df[c].median()), float(df[c].mean())
    (out / f"{stem}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
