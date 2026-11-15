"""Model size and compute per STFT frame (multiply-accumulates, counted from the layer shapes)."""
from __future__ import annotations

import torch

from dwmss.model import MaskNet, count_parameters


def macs_per_frame(model: MaskNet, n_bins: int) -> dict:
    """MACs for one frame of n_bins frequency bins, split by stage (conv / pool+in_proj / gru / head)."""
    c1, c2 = model.conv1.out_channels, model.conv2.out_channels
    with torch.no_grad():
        L2 = model.conv2(model.conv1(torch.zeros(1, 1, n_bins))).shape[-1]
        L3 = model.conv3(model.conv2(model.conv1(torch.zeros(1, 1, n_bins)))).shape[-1]
    H, hh = model.gru.hidden_size, model.head_g.out_features
    parts = {
        "conv": n_bins * c1 * model.conv1.kernel_size[0] + L2 * c2 * c1 * model.conv2.kernel_size[0]
        + L3 * c2 * c2 * model.conv3.kernel_size[0],
        "pool+in_proj": L3 * model.n_bands * c2 + c2 * model.n_bands * H,
        "gru": 6 * H * H,
        "head": H * hh + n_bins * c1 * hh + n_bins * hh,
    }
    parts["total"] = sum(parts.values())
    return parts


def describe(cfg: dict, sr: int = 44100) -> dict:
    """Parameters and compute of a config (no weights needed)."""
    model = MaskNet(**cfg["model"])
    n_bins = cfg["stft"]["n_analysis"] // 2 + 1
    hop = cfg["stft"]["n_synthesis"] // 2
    macs = macs_per_frame(model, n_bins)
    return {"params": count_parameters(model), "hidden": model.gru.hidden_size, "n_bins": n_bins,
            "hop_samples": hop, "algorithmic_latency_ms": 1e3 * cfg["stft"]["n_synthesis"] / sr,
            "macs_per_frame_M": macs["total"] / 1e6, "gmacs_per_s": macs["total"] * sr / hop / 1e9}
