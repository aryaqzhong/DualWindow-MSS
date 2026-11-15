"""Vocal bleed & preservation benchmark: does lowering latency cost more vocal preservation or more
bleed suppression?

Per test song: the 30 s excerpt with the most vocal energy, preceded by 5 s of real context that only
warms up the recurrent state (not scored). Inputs (mono): vocals only; accompaniment only
(drums + bass + other); vocals + accompaniment scaled to a bleed-to-vocal ratio (BVR) of −15, −5, +5 dB.
The model computes its mask from each input. Because synthesis is linear, applying that mask to the
vocal and to each accompaniment stem separately splits the output exactly into preserved vocal and
residual bleed. Mask strength is swept as M**gamma (gamma 0.5–2) for vocals-only and the −5 dB mix.

Usage: python -I scripts/bleed_benchmark.py [--runs runs/A_sym1024_seed0 ...] [--max-songs K] [--device cpu]
Outputs: results/bleed/<run>.csv, results/bleed/summary.md, results/bleed/fig_bleed_*.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.signal import welch  # noqa: E402

from dwmss.data import SR, STEMS, StemStore, list_tracks  # noqa: E402
from dwmss.metrics import si_sdr  # noqa: E402
from dwmss.model import MaskNet  # noqa: E402
from dwmss.plotstyle import (AXIS, CONFIG_COLOR, CONFIG_LABEL, CONFIG_SHORT, INK, INK2, apply_style, dot,  # noqa: E402
                             header, plt)
from dwmss.stft import DualWindowSTFT  # noqa: E402

DEFAULT_RUNS = ["runs/A_sym1024_seed0", "runs/A_sym1024_seed1", "runs/B_sym256_seed0", "runs/B_sym256_seed1",
                "runs/C_dual_seed0", "runs/C_dual_seed1", "runs/D_dual_2p9ms_seed0", "runs/D_dual_2p9ms_seed1",
                "runs/E_sym128_seed0", "runs/E_sym128_seed1"]
BVRS = [-15.0, -5.0, 5.0]
GAMMAS = [0.5, 0.75, 1.0, 1.5, 2.0]
SWEEP_BVR = -5.0
OCTAVES = [(125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
EXCERPT_S, CONTEXT_S = 30.0, 5.0


def expected_conditions() -> set[tuple[str, float]]:
    """The complete protocol for one song (13 distinct condition/gamma rows)."""
    return ({("vocal_only", g) for g in GAMMAS} | {("instr_only", 1.0)} |
            {(f"mix_bvr{bvr:+.0f}", g) for bvr in BVRS
             for g in (GAMMAS if bvr == SWEEP_BVR else [1.0])})


def cached_run_complete(path: Path, songs, run_name: str, config_name: str) -> bool:
    """Accept a cache only when every requested song has the complete scored protocol.

    A truncated CSV, repeated row, different smoke-test subset, wrong run/config or missing
    measurement must be recomputed. Structural NaNs in columns unused by a condition are allowed.
    """
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    keys = ["song", "condition", "gamma"]
    if not set(keys + ["run", "config"]).issubset(df.columns):
        return False
    if set(df["run"]) != {run_name} or set(df["config"]) != {config_name}:
        return False
    expected = {(song, condition, gamma) for song in songs for condition, gamma in expected_conditions()}
    if len(df) != len(expected) or df.duplicated(keys).any():
        return False
    observed = set(df[keys].itertuples(index=False, name=None))
    if observed != expected:
        return False
    per_stem = [f"atten_{s}_db" for s in STEMS[1:]]
    for condition, group in df.groupby("condition"):
        if condition == "vocal_only":
            columns = ["vocal_loss_db", "si_sdr"]
        elif condition == "instr_only":
            columns = ["bleed_atten_db"] + per_stem
        else:
            columns = ["si_sdr", "si_sdr_input", "vocal_preserved_si_sdr", "vocal_loss_db", "bleed_atten_db",
                       "bvr_db"] + per_stem
            if "bvr_db" not in group or not np.allclose(pd.to_numeric(group["bvr_db"], errors="coerce"),
                                                       float(condition.removeprefix("mix_bvr"))):
                return False
        if not set(columns).issubset(df.columns):
            return False
        if not np.isfinite(group[columns].apply(pd.to_numeric, errors="coerce").to_numpy()).all():
            return False
        if condition.startswith("mix_bvr"):
            octave_columns = [f"loss_{a}_{b}" for a, b in OCTAVES]
            if not set(octave_columns).issubset(df.columns):
                return False
            if not np.isfinite(group.loc[group.gamma == 1.0, octave_columns].apply(
                    pd.to_numeric, errors="coerce").to_numpy()).all():
                return False
    return True


def common_attenuation(curves: dict[str, pd.DataFrame]):
    """Interpolate vocal loss only inside every curve's attainable attenuation range.

    Returns (target, losses), or None if there is no common operating point. In particular,
    callers must not reuse the old A/B/C target for D/E without checking their ranges.
    """
    if not curves:
        return None
    clean = {}
    for name, curve in curves.items():
        xy = curve[["bleed_atten_db", "vocal_loss_db"]].dropna()
        xy = xy[np.isfinite(xy).all(axis=1)].groupby("bleed_atten_db", as_index=False)["vocal_loss_db"].mean()
        if xy.empty:
            return None
        clean[name] = xy.sort_values("bleed_atten_db")
    lo = max(curve.bleed_atten_db.min() for curve in clean.values())
    hi = min(curve.bleed_atten_db.max() for curve in clean.values())
    if hi < lo:
        return None
    target = float((lo + hi) / 2)
    return target, {name: float(np.interp(target, curve.bleed_atten_db, curve.vocal_loss_db))
                    for name, curve in clean.items()}


def energy(x):
    return float(np.sum(np.asarray(x, np.float64) ** 2)) + 1e-12


def db(r):
    return float(10 * np.log10(r))


def signed(v: float) -> str:
    """'−5' with a typographic minus, for figure text."""
    return f"{v:+.0f}".replace("-", "−")


def pick_excerpt(voc: np.ndarray) -> int:
    blk, k = SR, int(EXCERPT_S)
    e = np.array([np.sum(voc[i * blk:(i + 1) * blk] ** 2) for i in range(len(voc) // blk)])
    cs = np.convolve(e, np.ones(k), mode="valid")
    lo = int(CONTEXT_S) + 1
    hi = max(lo + 1, len(cs) - 1)
    return (lo + int(np.argmax(cs[lo:hi]))) * blk  # start of the scored region


@torch.no_grad()
def mask_of(model, stft, x, device, chunk=4096):
    X = stft.analysis(torch.tensor(x, device=device)[None])
    ms, h = [], None
    for s in range(0, X.shape[1], chunk):
        m, h = model(X[:, s:s + chunk], stft.window_energy, h)
        ms.append(m)
    return torch.cat(ms, dim=1)


@torch.no_grad()
def apply(stft, M, comp_spec, n):
    return stft.synthesis(M * comp_spec, n)[0].cpu().numpy()


def octave_loss(out, ref):
    f, po = welch(out, SR, nperseg=4096)
    _, pr = welch(ref, SR, nperseg=4096)
    return {f"loss_{a}_{b}": db((po[(f >= a) & (f < b)].sum() + 1e-12) / (pr[(f >= a) & (f < b)].sum() + 1e-12))
            for a, b in OCTAVES}


@torch.no_grad()
def run_one(run: Path, songs, store, device):
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    stft = DualWindowSTFT(cfg["stft"]["n_analysis"], cfg["stft"]["n_synthesis"]).to(device)
    model = MaskNet(**cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    rows = []
    for si, song in enumerate(songs):
        n_all = store.lengths[song]
        voc_full = store.read(song, "vocals", 0, n_all)
        s0 = pick_excerpt(voc_full)
        a0, L, C = s0 - int(CONTEXT_S * SR), int(EXCERPT_S * SR), int(CONTEXT_S * SR)
        n = C + L
        sc = slice(C, C + L)
        v = voc_full[a0:a0 + n]
        stems = {s: store.read(song, s, a0, n) for s in STEMS[1:]}
        acc = sum(stems.values())
        V = stft.analysis(torch.tensor(v, device=device)[None])
        S = {s: stft.analysis(torch.tensor(x, device=device)[None]) for s, x in stems.items()}
        base = {"song": song, "config": cfg["name"], "run": run.name}

        # vocals only (gamma sweep)
        Mv = mask_of(model, stft, v, device)
        for g in GAMMAS:
            out = apply(stft, Mv ** g, V, n)[sc]
            rows.append({**base, "condition": "vocal_only", "gamma": g,
                         "vocal_loss_db": db(energy(out) / energy(v[sc])), "si_sdr": si_sdr(out, v[sc])})
        # accompaniment only (gamma = 1), per-stem attenuation by decomposition
        Ma = mask_of(model, stft, acc, device)
        r = {**base, "condition": "instr_only", "gamma": 1.0}
        tot = 0.0
        for s in stems:
            o = apply(stft, Ma, S[s], n)[sc]
            r[f"atten_{s}_db"] = db(energy(stems[s][sc]) / energy(o))
            tot = tot + o
        r["bleed_atten_db"] = db(energy(acc[sc]) / energy(tot))
        rows.append(r)
        # mixtures at several bleed-to-vocal ratios
        for bvr in BVRS:
            gain = np.sqrt(energy(v[sc]) / energy(acc[sc]) * 10 ** (bvr / 10))
            mix = (v + gain * acc).astype(np.float32)
            Mm = mask_of(model, stft, mix, device)
            for g in (GAMMAS if bvr == SWEEP_BVR else [1.0]):
                M = Mm ** g
                out_v = apply(stft, M, V, n)[sc]
                out_s = {s: apply(stft, M, S[s], n)[sc] * gain for s in stems}
                out_a = sum(out_s.values())
                rr = {**base, "condition": f"mix_bvr{bvr:+.0f}", "bvr_db": bvr, "gamma": g,
                      "si_sdr": si_sdr(out_v + out_a, v[sc]), "si_sdr_input": si_sdr(mix[sc], v[sc]),
                      "vocal_preserved_si_sdr": si_sdr(out_v, v[sc]),
                      "vocal_loss_db": db(energy(out_v) / energy(v[sc])),
                      "bleed_atten_db": db(energy(gain * acc[sc]) / energy(out_a))}
                for s in stems:
                    rr[f"atten_{s}_db"] = db(energy(gain * stems[s][sc]) / energy(out_s[s]))
                if g == 1.0:
                    rr.update(octave_loss(out_v, v[sc]))
                rows.append(rr)
        print(f"  [{si + 1}/{len(songs)}] {song}", flush=True)
    return pd.DataFrame(rows)


def summarize(out_dir: Path):
    frames = [pd.read_csv(f) for f in sorted(out_dir.glob("*.csv"))
              if f.name not in {"summary.csv", "equal_attenuation.csv"}]
    if not frames:
        return
    df = pd.concat(frames)
    # seed-average per song, then median over songs
    key = ["config", "song", "condition", "gamma"]
    num = [c for c in df.columns if c not in key + ["run", "bvr_db"] and df[c].dtype.kind in "fi"]
    per_song = df.groupby(key)[num].mean().reset_index()
    med = per_song.groupby(["config", "condition", "gamma"])[num].median().reset_index()
    med.to_csv(out_dir / "summary.csv", index=False)
    cfgs = sorted(med["config"].unique())
    n_songs = df["song"].nunique()
    apply_style()

    L = ["# Vocal bleed & preservation benchmark (MUSDB18-HQ test, 30 s vocal-rich excerpts, mono)", "",
         "Medians over songs of seed-averaged values. Vocal loss: output vocal energy relative to the clean vocal "
         "(dB, 0 = nothing lost). Bleed attenuation: accompaniment energy removed (dB, higher = less bleed).", "",
         f"Songs: {n_songs}. Distinct training runs per config: " + ", ".join(
             f"{c}: {df.loc[df.config == c, 'run'].nunique()}" for c in cfgs) + ".", "",
         "| config | vocals only: loss (dB) / SI-SDR | accompaniment only: attenuation (dB) | per stem (drums / bass / other) | "
         "mix −5 dB: SI-SDR / preserved-vocal SI-SDR / vocal loss / bleed atten. |",
         "|---|---|---:|---|---|"]
    for c in cfgs:
        m = med[med.config == c]
        vo = m[(m.condition == "vocal_only") & (m.gamma == 1.0)].iloc[0]
        io = m[m.condition == "instr_only"].iloc[0]
        mx = m[(m.condition == "mix_bvr-5") & (m.gamma == 1.0)].iloc[0]
        L.append(f"| {c} | {vo['vocal_loss_db']:+.2f} / {vo['si_sdr']:.1f} | {io['bleed_atten_db']:.1f} | "
                 f"{io['atten_drums_db']:.1f} / {io['atten_bass_db']:.1f} / {io['atten_other_db']:.1f} | "
                 f"{mx['si_sdr']:.2f} / {mx['vocal_preserved_si_sdr']:.1f} / {mx['vocal_loss_db']:+.2f} / {mx['bleed_atten_db']:.1f} |")
    L += ["", "## SI-SDR across bleed levels (gamma = 1)", "", "| config | " + " | ".join(f"BVR {b:+.0f} dB" for b in BVRS) + " |",
          "|---|" + "---:|" * len(BVRS)]
    for c in cfgs:
        m = med[(med.config == c) & (med.gamma == 1.0)]
        L.append(f"| {c} | " + " | ".join(f"{m[m.condition == f'mix_bvr{b:+.0f}']['si_sdr'].iloc[0]:.2f}" for b in BVRS) + " |")

    # preservation-suppression curves at the sweep BVR
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    fig.subplots_adjust(left=0.11, right=0.80, top=0.80, bottom=0.12)
    curves = {}
    for c in cfgs:
        m = med[(med.config == c) & (med.condition == f"mix_bvr{SWEEP_BVR:+.0f}")].sort_values("gamma")
        curves[c] = m
        ax.plot(m["bleed_atten_db"], m["vocal_loss_db"], color=CONFIG_COLOR.get(c, AXIS), lw=2, zorder=2,
                linestyle="--" if c in {"D_dual_2p9ms", "E_sym128"} else "-",
                label=CONFIG_LABEL.get(c, c))
        dot(ax, m["bleed_atten_db"], m["vocal_loss_db"], CONFIG_COLOR.get(c, AXIS), size=46, zorder=3)
        end = m.iloc[-1]  # strongest mask: direct label at the curve end
        ax.annotate(CONFIG_SHORT.get(c, c), (end["bleed_atten_db"], end["vocal_loss_db"]), xytext=(8, -2),
                    textcoords="offset points", fontsize=10.5, fontweight="bold", color=INK, va="center")
    ax.margins(x=0.08, y=0.1)
    ax.axhline(0, color=AXIS, lw=1, zorder=0)
    ax.grid(True)
    ax.set_xlabel("Bleed attenuation (dB, higher = less accompaniment leaks through)")
    ax.set_ylabel("Vocal loss (dB, 0 = vocal fully kept)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    header(fig, 0.01, 0.80, "Keeping the vocal vs. removing the bleed",
           f"Simulated vocal-mic bleed at {signed(SWEEP_BVR)} dB (accompaniment relative to vocal) · each curve sweeps "
           f"the mask strength γ\nfrom {GAMMAS[0]:g} (left end, gentle) to {GAMMAS[-1]:g} (right end, strong) · "
           f"up and to the right is better · medians over {n_songs} test songs, seed-averaged")
    fig.savefig(out_dir / "fig_bleed_tradeoff.png")
    plt.close(fig)
    # Preserve the original A/B/C comparison explicitly; D/E and the full set get their
    # own attainable targets rather than extrapolating the old ~9.7 dB operating point.
    scopes = []
    abc = ["A_sym1024", "B_sym256", "C_dual"]
    de = ["E_sym128", "D_dual_2p9ms"]
    if set(abc).issubset(curves):
        scopes.append(("A/B/C reference", abc))
    if set(cfgs) != set(abc):
        scopes.append(("All available configurations", cfgs))
    if set(de).issubset(curves) and set(cfgs) != set(de):
        scopes.append(("D/E at 2.9 ms", de))
    L += ["", "## Vocal loss at equal bleed attenuation", "",
          f"BVR {SWEEP_BVR:+.0f} dB. Each block uses the midpoint of its curves' common attainable attenuation "
          "range, interpolated along the gamma sweep. Targets are recomputed for each comparison; "
          "losses in blocks with different targets are not directly comparable."]
    equal_rows = []
    for scope, names in scopes:
        result = common_attenuation({c: curves[c] for c in names})
        if result is None:
            L += ["", f"### {scope}", "", "No common attainable attenuation; no extrapolated comparison is reported."]
            continue
        target, losses = result
        L += ["", f"### {scope} ({target:.1f} dB attenuation)", "",
              "| config | vocal loss (dB) |", "|---|---:|"]
        for c in names:
            L.append(f"| {c} | {losses[c]:+.2f} |")
            equal_rows.append({"comparison": scope, "config": c, "bleed_atten_db": target,
                               "vocal_loss_db": losses[c]})
    pd.DataFrame(equal_rows, columns=["comparison", "config", "bleed_atten_db", "vocal_loss_db"]).to_csv(
        out_dir / "equal_attenuation.csv", index=False)

    # per-octave vocal loss
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    fig.subplots_adjust(left=0.11, right=0.80, top=0.78, bottom=0.13)
    oc = [f"loss_{a}_{b}" for a, b in OCTAVES]
    ends = {}
    for c in cfgs:
        m = med[(med.config == c) & (med.condition == f"mix_bvr{SWEEP_BVR:+.0f}") & (med.gamma == 1.0)].iloc[0]
        vals = [m[k] for k in oc]
        ax.plot(range(len(oc)), vals, color=CONFIG_COLOR.get(c, AXIS), lw=2, zorder=2, label=CONFIG_LABEL.get(c, c),
                linestyle="--" if c in {"D_dual_2p9ms", "E_sym128"} else "-")
        dot(ax, range(len(oc)), vals, CONFIG_COLOR.get(c, AXIS), size=46, zorder=3)
        ends[c] = vals
    # direct labels at the end (first or last band) where the curves are farthest apart
    gap = {i: min((abs(a[i] - b[i]) for a, b in itertools.combinations(ends.values(), 2)), default=np.inf)
           for i in (0, len(oc) - 1)}
    i = max(gap, key=gap.get)
    ax.set_xlim(-0.6, len(oc) - 0.4)
    ax.set_xticks(range(len(oc)), [f"{a}–{b}" if b < 1000 else f"{a / 1000:g}–{b / 1000:g}k" for a, b in OCTAVES])
    ax.axhline(0, color=AXIS, lw=1, zorder=0)
    # keep the end labels at least 14 pt apart so that close curves (e.g. A and D) do not print on top of each other
    lo, hi = ax.get_ylim()
    min_sep = (hi - lo) / (fig.get_figheight() * 72 * ax.get_position().height) * 14
    order = sorted(ends, key=lambda c: ends[c][i])
    label_y = [ends[c][i] for c in order]
    for _ in range(100):
        moved = False
        for k in range(1, len(label_y)):
            short = min_sep - (label_y[k] - label_y[k - 1])
            if short > 1e-9:
                label_y[k - 1] -= short / 2
                label_y[k] += short / 2
                moved = True
        if not moved:
            break
    for c, y in zip(order, label_y):
        ax.annotate(CONFIG_SHORT.get(c, c), (i, y), xytext=(-10 if i == 0 else 10, 0), textcoords="offset points",
                    fontsize=10.5, fontweight="bold", color=INK, va="center", ha="right" if i == 0 else "left")
    ax.set_ylim(lo, hi)
    ax.yaxis.grid(True)
    ax.set_xlabel("Octave band (Hz)")
    ax.set_ylabel("Vocal loss (dB, 0 = fully kept)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    header(fig, 0.01, 0.78, "Where in the spectrum the vocal is lost",
           f"Preserved vocal vs. clean vocal per octave · bleed {signed(SWEEP_BVR)} dB, mask strength γ = 1 · "
           f"medians over {n_songs} test songs · dashed lines: 2.9 ms")
    fig.savefig(out_dir / "fig_bleed_octaves.png")
    plt.close(fig)
    L += ["", f"Figures: `{out_dir.name}/fig_bleed_tradeoff.png`, `{out_dir.name}/fig_bleed_octaves.png`. Simulated bleed (digital "
              "mixing of studio stems); not a real stage microphone and not acoustic feedback."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--max-songs", type=int)
    ap.add_argument("--device")
    ap.add_argument("--musdb-root", default="data/musdb18hq")
    ap.add_argument("--out", default=str(REPO / "results" / "bleed"))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a run CSV only if every requested song/condition/gamma and scored metric is complete")
    args = ap.parse_args()
    if args.max_songs is not None and args.max_songs <= 0:
        ap.error("--max-songs must be positive")
    out = Path(args.out)
    if args.max_songs is not None and out.resolve() == (REPO / "results" / "bleed").resolve():
        ap.error("use a separate --out directory for --max-songs so full-test results are preserved")
    run_paths = [p if p.is_absolute() else REPO / p for p in map(Path, args.runs)]
    if len({p.name for p in run_paths}) != len(run_paths):
        ap.error("run directory names must be unique")
    out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        songs = list_tracks(args.musdb_root, "test")
        if args.max_songs is None and len(songs) != 50:
            raise ValueError(f"full protocol expects 50 test songs, found {len(songs)}")
        songs = songs[:args.max_songs] if args.max_songs else songs
        store = None
        for run in run_paths:
            if not (run / "best.pt").exists():
                raise FileNotFoundError(f"requested run has no checkpoint: {run / 'best.pt'}")
            meta = run / "meta.json"
            cfg_name = (json.loads(meta.read_text())["config"]["name"] if meta.exists() else
                        torch.load(run / "best.pt", map_location="cpu", weights_only=False)["config"]["name"])
            path = out / f"{run.name}.csv"
            if args.skip_existing and cached_run_complete(path, songs, run.name, cfg_name):
                print(f"=== reuse {run.name}: complete protocol for {len(songs)} songs", flush=True)
                continue
            if store is None:
                store = StemStore(args.musdb_root, "test", songs)
            print(f"=== {run.name}", flush=True)
            frame = run_one(run, songs, store, device)
            temp = path.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            if not cached_run_complete(temp, songs, run.name, cfg_name):
                raise RuntimeError(f"incomplete benchmark output: {temp}")
            temp.replace(path)
    summarize(out)


if __name__ == "__main__":
    main()
