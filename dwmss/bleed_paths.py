"""Fixed, causal accompaniment paths for a frozen-model robustness experiment.

These are synthetic feed-forward FIR stress tests, not measured rooms, microphone
models, or acoustic-feedback simulation. Parameters are fixed before evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.signal import firwin, oaconvolve


@dataclass(frozen=True)
class BleedPath:
    name: str
    description: str
    impulse: np.ndarray
    parameters: dict

    def provenance(self) -> dict:
        h = np.asarray(self.impulse, dtype="<f8")
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters, "samples": len(h),
                "sha256_float64_le": hashlib.sha256(h.tobytes()).hexdigest(),
                "coefficients": h.tolist()}


def fixed_paths(sr: int = 44100) -> dict[str, BleedPath]:
    """Three predeclared paths; the same impulse is applied to every stem/song."""
    if sr != 44100:
        raise ValueError("the frozen protocol is defined at 44100 Hz")
    lowpass = firwin(129, 6500.0, fs=sr)
    mild = 0.35 * lowpass
    mild[64] += 0.65  # unity DC; high frequencies approach -3.74 dB
    delay = round(0.004 * sr)
    mild = np.pad(mild, (delay, 0))

    seed, onset, tail_s, rt60 = 20261109, 0.012, 0.180, 0.120
    onset_n = round(onset * sr)
    tail_n = round(tail_s * sr)
    room = np.zeros(onset_n + tail_n + 1, dtype=np.float64)
    room[onset_n] = 1.0
    reflections = [(0.017, 0.35), (0.031, -0.20), (0.047, 0.14)]
    for seconds, amplitude in reflections:
        room[onset_n + round(seconds * sr)] += amplitude
    rng = np.random.default_rng(seed)
    diffuse = rng.standard_normal(tail_n)
    diffuse = oaconvolve(diffuse, firwin(65, 4500.0, fs=sr))[:tail_n]
    diffuse *= np.exp(-np.log(1000.0) * np.arange(tail_n) / (sr * rt60))
    diffuse *= 0.18 / np.linalg.norm(diffuse)
    room[onset_n + 1:onset_n + 1 + tail_n] += diffuse
    return {
        "dry": BleedPath("dry", "Identity path (digital-mixing control).", np.ones(1), {}),
        "filtered_delay": BleedPath(
            "filtered_delay", "Mild fixed spectral tilt and causal propagation delay.", mild,
            {"onset_delay_s": 0.004, "onset_delay_samples": delay, "fir_taps": 129,
             "lowpass_cutoff_hz": 6500.0, "lowpass_weight": 0.35,
             "center_impulse_weight": 0.65, "filter_group_delay_samples": 64}),
        "synthetic_room": BleedPath(
            "synthetic_room", "Fixed direct path, three reflections and a finite seeded diffuse tail.", room,
            {"onset_delay_s": onset, "onset_delay_samples": onset_n,
             "tail_duration_s": tail_s, "diffuse_decay_rt60_s": rt60,
             "diffuse_l2_relative_to_direct": 0.18, "diffuse_lowpass_hz": 4500.0,
             "diffuse_filter_taps": 65, "seed": seed,
             "reflections_after_direct_s_and_amplitude": reflections}),
    }


def causal_filter(x: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Zero-state causal convolution, same input length, without delay compensation."""
    x, h = np.asarray(x), np.asarray(impulse, dtype=np.float64)
    if x.ndim != 1 or h.ndim != 1 or not len(h):
        raise ValueError("expected a mono signal and a nonempty one-dimensional FIR")
    if not np.isfinite(x).all() or not np.isfinite(h).all():
        raise ValueError("signal and FIR must be finite")
    if not len(x):
        return np.empty(0, dtype=np.float32)
    return oaconvolve(x.astype(np.float64), h, mode="full")[:len(x)].astype(np.float32)


def filtered_window(x_with_history: np.ndarray, impulse: np.ndarray, offset: int,
                    length: int) -> np.ndarray:
    """Filter real prehistory first, then cut the model's context/scored window.

    Callers read at least len(impulse)-1 preceding samples, unless the song starts
    sooner. Cutting before convolution would reset the path at the context edge.
    """
    if offset < 0 or length < 0 or offset + length > len(x_with_history):
        raise ValueError("window lies outside the supplied history and signal")
    return causal_filter(x_with_history, impulse)[offset:offset + length]


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def scale_to_bvr(vocal: np.ndarray, stems: dict[str, np.ndarray], scored: slice,
                 bvr_db: float = -5.0) -> tuple[dict[str, np.ndarray], float, float]:
    """One gain on all transformed stems, calibrated on the scored window only."""
    if not stems or any(np.shape(s) != np.shape(vocal) for s in stems.values()):
        raise ValueError("nonempty stems must have the same mono shape as the vocal")
    acc = sum(stems.values())
    ev, ea = energy(vocal[scored]), energy(acc[scored])
    if ev <= 1e-20 or ea <= 1e-20 or not np.isfinite(ev + ea + bvr_db):
        raise ValueError("cannot calibrate BVR with silent/nonfinite vocal or accompaniment")
    gain = float(np.sqrt(10 ** (bvr_db / 10) * ev / ea))
    scaled = {s: (gain * x).astype(np.float32) for s, x in stems.items()}
    achieved = float(10 * np.log10(energy(sum(scaled.values())[scored]) / ev))
    if abs(achieved - bvr_db) > 1e-4:
        raise AssertionError(f"BVR normalization failed: {achieved} vs {bvr_db}")
    return scaled, gain, achieved
