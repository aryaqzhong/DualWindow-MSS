"""Lightweight causal mask estimator whose parameter count does not depend on the FFT size.

Per frame: log power spectrum -> frequency-direction Conv1d encoder (weights shared across
bins) -> adaptive pooling to a fixed number of bands -> 1-layer unidirectional GRU over time
-> shared per-bin head (conv feature of the bin + frequency position + GRU state) -> sigmoid
mask. Nothing mixes information across frames except the causal GRU, so processing a
sequence in chunks while carrying the GRU state gives the same output as one pass.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def freq_position_features(n_bins: int, n_feats: int = 8, device=None) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, n_bins, device=device)
    feats = []
    for j in range(n_feats // 2):
        feats += [torch.sin(math.pi * (2 ** j) * u), torch.cos(math.pi * (2 ** j) * u)]
    return torch.stack(feats, dim=-1)  # (F, n_feats)


class MaskNet(nn.Module):
    def __init__(self, n_bands=64, c1=16, c2=32, hidden=256, head_hidden=32, n_pos=8,
                 feat_offset=-5.0, feat_scale=5.0):
        super().__init__()
        self.n_bands, self.n_pos = int(n_bands), int(n_pos)
        self.feat_offset, self.feat_scale = float(feat_offset), float(feat_scale)
        self.conv1 = nn.Conv1d(1, c1, 9, padding=4)
        self.conv2 = nn.Conv1d(c1, c2, 7, stride=2, padding=3)
        self.conv3 = nn.Conv1d(c2, c2, 7, stride=2, padding=3)
        self.in_proj = nn.Linear(c2 * self.n_bands, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        # Head first layer is factorized: the GRU term is computed once per frame and
        # broadcast over bins; only the conv and position terms are computed per bin.
        self.head_g = nn.Linear(hidden, head_hidden)
        self.head_c = nn.Linear(c1, head_hidden, bias=False)
        self.head_p = nn.Linear(self.n_pos, head_hidden, bias=False)
        self.head_out = nn.Linear(head_hidden, 1)
        self._pos = {}

    def _positions(self, n_bins: int, device) -> torch.Tensor:
        key = (n_bins, str(device))
        if key not in self._pos:
            self._pos[key] = freq_position_features(n_bins, self.n_pos, device)
        return self._pos[key]

    def features(self, X: torch.Tensor, window_energy: float) -> torch.Tensor:
        p = X.real ** 2 + X.imag ** 2
        return (torch.log(p / window_energy + 1e-10) - self.feat_offset) / self.feat_scale

    def forward(self, X: torch.Tensor, window_energy: float, h0=None):
        """X: (B, T, F) complex STFT. Returns mask (B, T, F) in [0, 1] and the GRU state."""
        B, T, n_bins = X.shape
        z = self.features(X, window_energy).reshape(B * T, 1, n_bins)
        f1 = F.gelu(self.conv1(z))  # (BT, c1, F)
        f3 = F.gelu(self.conv3(F.gelu(self.conv2(f1))))  # (BT, c2, ~F/4)
        pooled = F.adaptive_avg_pool1d(f3, self.n_bands).reshape(B, T, -1)
        g, h_n = self.gru(self.in_proj(pooled), h0)  # (B, T, hidden)
        h = (
            self.head_c(f1.transpose(1, 2)).reshape(B, T, n_bins, -1)
            + self.head_p(self._positions(n_bins, X.device))
            + self.head_g(g).unsqueeze(2)
        )
        mask = torch.sigmoid(self.head_out(F.gelu(h))).squeeze(-1)
        return mask, h_n


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
