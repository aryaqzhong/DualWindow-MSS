"""Signal metrics (float64, whole-signal)."""
from __future__ import annotations

import numpy as np


def si_sdr(est, ref, eps: float = 1e-10) -> float:
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def usdr(est, ref, eps: float = 1e-10) -> float:
    """Utterance-level SDR as used in the Music Demixing challenges: 10 log10(|s|^2 / |s - s_hat|^2)."""
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(10.0 * np.log10((np.dot(ref, ref) + eps) / (np.dot(ref - est, ref - est) + eps)))
