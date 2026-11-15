"""Per-frame model step with ONNX-friendly operations, for CPU streaming benchmarks.

Identical to MaskNet applied to one frame, except that adaptive average pooling is written as a
fixed matrix (same bins as torch.nn.functional.adaptive_avg_pool1d) and the input is the log power
spectrum of the frame (the STFT stays outside the model).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dwmss.model import MaskNet, freq_position_features


def adaptive_pool_matrix(length: int, out: int) -> torch.Tensor:
    P = torch.zeros(length, out)
    for i in range(out):
        a, b = (i * length) // out, math.ceil((i + 1) * length / out)
        P[a:b, i] = 1.0 / (b - a)
    return P


class FrameStep(nn.Module):
    def __init__(self, model: MaskNet, n_bins: int):
        super().__init__()
        self.m = model.eval()
        with torch.no_grad():
            L = model.conv3(model.conv2(model.conv1(torch.zeros(1, 1, n_bins)))).shape[-1]
            pos = model.head_p(freq_position_features(n_bins, model.n_pos))
        self.register_buffer("P", adaptive_pool_matrix(L, model.n_bands))
        self.register_buffer("pos_term", pos)

    def forward(self, logpow: torch.Tensor, h: torch.Tensor):
        """logpow: (1, F) log(|X|^2 / window_energy + 1e-10); h: (1, 1, hidden). Returns mask (1, F), h."""
        m = self.m
        z = ((logpow - m.feat_offset) / m.feat_scale).unsqueeze(1)
        f1 = F.gelu(m.conv1(z))
        f3 = F.gelu(m.conv3(F.gelu(m.conv2(f1))))
        pooled = torch.matmul(f3, self.P).reshape(1, 1, -1)
        g, h_n = m.gru(m.in_proj(pooled), h)
        hsum = m.head_c(f1.transpose(1, 2)) + self.pos_term + m.head_g(g[:, 0]).unsqueeze(1)
        return torch.sigmoid(m.head_out(F.gelu(hsum))).squeeze(-1), h_n


def export_onnx(model: MaskNet, n_bins: int, path: str, opset: int = 17) -> None:
    step = FrameStep(model, n_bins)
    hidden = model.gru.hidden_size
    torch.onnx.export(step, (torch.zeros(1, n_bins), torch.zeros(1, 1, hidden)), path,
                      input_names=["logpow", "h"], output_names=["mask", "h_out"], opset_version=opset,
                      dynamo=False)


class OnnxStep:
    """Callable(logpow (F,), h) -> (mask (F,), h) backed by ONNX Runtime with a fixed thread count."""

    def __init__(self, path: str, hidden: int, threads: int = 1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.h0 = np.zeros((1, 1, hidden), dtype=np.float32)

    def __call__(self, logpow: np.ndarray, h):
        mask, h_out = self.sess.run(None, {"logpow": logpow[None], "h": self.h0 if h is None else h})
        return mask[0], h_out
