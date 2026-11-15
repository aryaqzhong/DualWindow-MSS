"""Oracle-mask reference on the 14 musdb validation tracks; selects the analysis window for C.

Protocol (same as pilot/oracle_dualwin.py): mono, the 30 s excerpt with the most vocal energy
per track (8192 samples of real context on each side, excluded from scoring), mixture = sum of
stems, IRM1 = |V| / (|V| + |A|), IRM2 = |V|^2 / (|V|^2 + |A|^2), mixture phase, uSDR.
Uses the same DualWindowSTFT module as training.

Usage: python -I scripts/oracle_select.py [--subset valid|test] [--out results/oracle_validation]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dwmss.config import load_config  # noqa: E402
from dwmss.data import SR, STEMS, StemStore, list_tracks, train_valid_split  # noqa: E402
from dwmss.metrics import usdr  # noqa: E402
from dwmss.stft import DualWindowSTFT  # noqa: E402

CONFIGS = [
    ("sym1024", 1024, 1024), ("sym512", 512, 512), ("sym256", 256, 256), ("sym128", 128, 128),
    ("dual1024/256", 1024, 256), ("dual2048/256", 2048, 256), ("dual4096/256", 4096, 256),
    ("dual1024/128", 1024, 128), ("dual2048/128", 2048, 128),
]
CANDIDATES_FOR_C = ["dual1024/256", "dual2048/256", "dual4096/256"]


def excerpt(voc: np.ndarray, seconds: float, context: int) -> int:
    blk = SR
    e = np.array([np.sum(voc[i * blk:(i + 1) * blk] ** 2) for i in range(len(voc) // blk)])
    k = int(seconds)
    cs = np.convolve(e, np.ones(k), mode="valid")
    lo, hi = context // blk + 1, max(context // blk + 2, len(cs) - (context // blk + 2))
    b = lo + int(np.argmax(cs[lo:hi]))
    return b * blk - context


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--subset", choices=["valid", "test"], default="valid")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=str(REPO / "results" / "oracle_validation"))
    args = ap.parse_args()
    torch.set_num_threads(4)

    root = load_config(args.config)["data"]["musdb_root"]
    if args.subset == "valid":
        tracks, folder = train_valid_split(root)[1], "train"
    else:
        tracks, folder = list_tracks(root, "test"), "test"
    store = StemStore(root, folder, tracks)
    stfts = {name: DualWindowSTFT(N, M) for name, N, M in CONFIGS}
    L, C = int(args.seconds * SR), 8192
    res = {name: {"irm1": [], "irm2": []} for name, _, _ in CONFIGS}
    for i, t in enumerate(tracks):
        n = store.lengths[t]
        voc_full = store.read(t, "vocals", 0, n)
        s0 = excerpt(voc_full, args.seconds, C)
        voc = voc_full[s0:s0 + L + 2 * C]
        acc = sum(store.read(t, s, s0, L + 2 * C) for s in STEMS[1:])
        v, a = torch.tensor(voc)[None], torch.tensor(acc)[None]
        for name, _, _ in CONFIGS:
            st = stfts[name]
            V, A = st.analysis(v), st.analysis(a)
            X = V + A
            pv, pa = V.abs(), A.abs()
            for key, m in (("irm1", pv / (pv + pa + 1e-8)), ("irm2", pv ** 2 / (pv ** 2 + pa ** 2 + 1e-8))):
                est = st.synthesis(m * X, v.shape[-1])[0].numpy()
                res[name][key].append(usdr(est[C:C + L], voc[C:C + L]))
        print(f"[{i + 1}/{len(tracks)}] {t}", flush=True)

    summary = []
    for name, N, M in CONFIGS:
        summary.append({
            "config": name, "N": N, "M": M, "latency_ms": round(1000 * M / SR, 2),
            "irm1_median": float(np.median(res[name]["irm1"])),
            "irm2_median": float(np.median(res[name]["irm2"])),
            "irm2_mean": float(np.mean(res[name]["irm2"])),
        })
    best = max((s for s in summary if s["config"] in CANDIDATES_FOR_C), key=lambda s: s["irm2_median"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"subset": args.subset, "tracks": tracks, "summary": summary,
                   "selected_for_C": best["config"], "per_track": res}, f, indent=1)
    lines = [f"# Oracle reference on the {args.subset} tracks ({len(tracks)} tracks)", "",
             f"Mono, {args.seconds:.0f} s max-vocal excerpts, uSDR of vocals (dB), mixture phase.", "",
             "| config | N | M | latency (ms) | IRM1 median | IRM2 median | IRM2 mean |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['config']} | {s['N']} | {s['M']} | {s['latency_ms']:.1f} | "
                     f"{s['irm1_median']:.2f} | {s['irm2_median']:.2f} | {s['irm2_mean']:.2f} |")
    lines += ["", f"Selected analysis window for C (highest IRM2 median among "
                  f"{', '.join(CANDIDATES_FOR_C)}): **{best['config']}** (N = {best['N']})."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
