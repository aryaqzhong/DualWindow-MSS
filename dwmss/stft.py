"""Causal symmetric / dual-window STFT analysis and synthesis (PyTorch, differentiable).

Window design (Mauler & Martin 2007 style; identical to ``pilot/oracle_dualwin.py``):

- analysis window ``wa`` (length N): sqrt of a long rising half-Hann over the first
  L = N - M/2 samples, then sqrt of the falling half of a periodic Hann(M);
- synthesis window ``ws``: nonzero only on the last M samples, ``ws = Hann(M) / wa``,
  so ``wa * ws = Hann(M)``, which overlap-adds to one at hop R = M/2.

Algorithmic latency = M samples (worst-case look-ahead M - 1). N == M gives the usual
sqrt-Hann / sqrt-Hann STFT.

Alignment: the signal is zero-padded by N samples on both sides and frame t covers padded
samples [t*R, t*R + N). Each frame ends at its newest sample, so no future input is used.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def make_windows(n_analysis: int, n_synthesis: int) -> tuple[np.ndarray, np.ndarray]:
    N, M = int(n_analysis), int(n_synthesis)
    if M <= 0 or M % 2 or N < M:
        raise ValueError(f"need even M > 0 and N >= M, got N={N}, M={M}")
    h = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(M) / M)  # periodic Hann: COLA at hop M/2
    L = N - M // 2
    wa = np.zeros(N)
    wa[:L] = np.sqrt(0.5 - 0.5 * np.cos(np.pi * np.arange(L) / L))
    wa[L:] = np.sqrt(h[M // 2:])
    ws = np.zeros(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = h / wa[N - M:]
    tail[~np.isfinite(tail)] = 0.0
    ws[N - M:] = tail
    return wa, ws


class DualWindowSTFT(torch.nn.Module):
    """Analysis with an N-sample window, synthesis with an M-sample window, hop M/2."""

    def __init__(self, n_analysis: int, n_synthesis: int):
        super().__init__()
        self.N, self.M = int(n_analysis), int(n_synthesis)
        self.R = self.M // 2
        wa, ws = make_windows(self.N, self.M)
        self.register_buffer("wa", torch.tensor(wa, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "ws_tail", torch.tensor(ws[self.N - self.M:], dtype=torch.float32), persistent=False
        )
        # Used to normalize magnitudes so features are comparable across window lengths.
        self.window_energy = float(np.sum(wa ** 2))

    @property
    def n_bins(self) -> int:
        return self.N // 2 + 1

    @property
    def hop(self) -> int:
        return self.R

    @property
    def latency_samples(self) -> int:
        return self.M

    def n_frames(self, length: int) -> int:
        return (length + self.N) // self.R + 1

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) real -> (B, T, F) complex."""
        xp = F.pad(x, (self.N, self.N))
        frames = xp.unfold(-1, self.N, self.R)
        return torch.fft.rfft(frames * self.wa, n=self.N, dim=-1)

    def synthesis(self, X: torch.Tensor, length: int) -> torch.Tensor:
        """X: (B, T, F) complex -> (B, length) real, aligned with the analysis input."""
        fr = torch.fft.irfft(X, n=self.N, dim=-1)[..., self.N - self.M:] * self.ws_tail
        B, T, _ = fr.shape
        R = self.R
        zero = fr.new_zeros(B, 1, R)
        blocks = torch.cat([fr[..., :R], zero], dim=1) + torch.cat([zero, fr[..., R:]], dim=1)
        seg = blocks.reshape(B, (T + 1) * R)  # padded sample indices [N - M, N - M + (T + 1) R)
        y = seg[:, self.M:self.M + length]  # original sample i sits at padded index i + N
        if y.shape[-1] < length:
            y = F.pad(y, (0, length - y.shape[-1]))
        return y
