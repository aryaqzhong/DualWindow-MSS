"""Reference-only activity and signal diagnostics for frozen-model failure analysis."""
from __future__ import annotations

import numpy as np

from dwmss.metrics import si_sdr, usdr


def block_power(x, block_samples):
    """Mean-square power per non-overlapping block, including a short final block."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or block_samples <= 0:
        raise ValueError("expected a nonempty finite mono signal and positive block size")
    starts = np.arange(0, len(x), block_samples)
    counts = np.minimum(block_samples, len(x) - starts)
    return np.add.reduceat(x * x, starts) / counts, counts


def power_db(power):
    return 10 * np.log10(np.maximum(np.asarray(power), 1e-24))


def activity_mask(reference, block_samples, relative_db=40.0, floor_dbfs=-60.0):
    """Select whole reference-vocal blocks above max(peak-relative_db, absolute floor)."""
    if relative_db <= 0 or not np.isfinite([relative_db, floor_dbfs]).all():
        raise ValueError("activity thresholds must be finite and relative_db positive")
    powers, counts = block_power(reference, block_samples)
    levels = power_db(powers)
    threshold = max(float(levels.max()) - relative_db, floor_dbfs)
    active_blocks = levels >= threshold
    selected = np.repeat(active_blocks, counts)
    return selected, threshold, levels


def projection_metrics(estimate, reference):
    """The projection used by SI-SDR, plus uncentered cosine and signal energies."""
    estimate, reference = np.asarray(estimate, np.float64), np.asarray(reference, np.float64)
    if estimate.shape != reference.shape or not len(reference):
        raise ValueError("projection requires equal nonempty signals")
    re = float(np.dot(reference, reference))
    ee = float(np.dot(estimate, estimate))
    cross = float(np.dot(estimate, reference))
    alpha = cross / (re + 1e-10)
    return {"si_sdr": si_sdr(estimate, reference), "usdr": usdr(estimate, reference),
            "alpha": alpha, "cosine": cross / np.sqrt(re * ee) if re * ee > 0 else None,
            "reference_energy": re, "estimate_energy": ee,
            "reference_dbfs": float(power_db(re / len(reference))),
            "estimate_dbfs": float(power_db(ee / len(reference)))}


def energy_db_ratio(numerator, denominator):
    a, b = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    ae, be = float(np.dot(a, a)), float(np.dot(b, b))
    return float(10 * np.log10(ae / be)) if ae > 0 and be > 0 else None


def diagnose_signals(vocal, accompaniment, estimate, preserved_vocal, residual_accompaniment,
                     block_samples=44100, relative_db=40.0, floor_dbfs=-60.0):
    arrays = [np.asarray(x) for x in (vocal, accompaniment, estimate, preserved_vocal, residual_accompaniment)]
    if any(x.shape != arrays[0].shape or x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("all diagnostic signals must be finite equal-length mono arrays")
    vocal, accompaniment, estimate, preserved_vocal, residual_accompaniment = arrays
    active, threshold, levels = activity_mask(vocal, block_samples, relative_db, floor_dbfs)
    inactive = ~active
    whole = projection_metrics(estimate, vocal)
    result = {f"whole_{k}": v for k, v in whole.items()}
    result.update({"relative_db": float(relative_db), "threshold_dbfs": threshold,
                   "floor_dbfs": float(floor_dbfs),
                   "active_samples": int(active.sum()), "inactive_samples": int(inactive.sum()),
                   "active_fraction": float(active.mean()), "peak_vocal_block_dbfs": float(levels.max()),
                   "vocal_to_accompaniment_db": energy_db_ratio(vocal, accompaniment)})
    result["active_reference_energy_fraction"] = (float(np.dot(np.asarray(vocal[active], np.float64),
                                                               np.asarray(vocal[active], np.float64)) /
                                                        whole["reference_energy"]) if whole["reference_energy"] > 0 else None)
    active_keys = list(whole) + ["vocal_loss_db", "accompaniment_attenuation_db", "preserved_vocal_si_sdr"]
    if active.any():
        result.update({f"active_{k}": v for k, v in projection_metrics(estimate[active], vocal[active]).items()})
        result.update({"active_vocal_loss_db": energy_db_ratio(preserved_vocal[active], vocal[active]),
                       "active_accompaniment_attenuation_db": energy_db_ratio(accompaniment[active], residual_accompaniment[active]),
                       "active_preserved_vocal_si_sdr": si_sdr(preserved_vocal[active], vocal[active])})
    else:
        result.update({f"active_{k}": None for k in active_keys})
    for name, x in (("output", estimate), ("residual_accompaniment", residual_accompaniment),
                    ("reference_vocal", vocal)):
        result[f"inactive_{name}_dbfs"] = (float(power_db(np.mean(np.asarray(x[inactive], np.float64) ** 2)))
                                               if inactive.any() else None)
    result["inactive_output_energy_fraction"] = (float(np.dot(np.asarray(estimate[inactive], np.float64),
                                                               np.asarray(estimate[inactive], np.float64)) /
                                                        (whole["estimate_energy"] + 1e-12)) if inactive.any() else None)
    return result


def paired_bootstrap(values, draws=10000):
    x = np.asarray(values, np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("paired bootstrap requires finite nonempty observations")
    samples = np.random.default_rng(0).choice(x, size=(draws, len(x)))
    median_ci = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    mean_ci = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return {"songs": len(x), "median": float(np.median(x)), "median_lo": float(median_ci[0]),
            "median_hi": float(median_ci[1]), "mean": float(x.mean()), "mean_lo": float(mean_ci[0]),
            "mean_hi": float(mean_ci[1]), "better": int((x > 0).sum()), "worst": float(x.min())}
