"""Training objective defined on the output waveform (identical for every window config)."""
from __future__ import annotations

import torch
import torch.nn as nn


class WaveMRSTFTLoss(nn.Module):
    """L1 on the waveform + L1 on power-compressed multi-resolution STFT magnitudes.

    The STFT settings here are fixed and independent of the separator's own analysis /
    synthesis windows, so configurations A/B/C all optimize the same objective.
    """

    def __init__(self, resolutions=((512, 128), (1024, 256), (2048, 512)), wave_weight=10.0,
                 power=0.3):
        super().__init__()
        self.resolutions = [(int(n), int(h)) for n, h in resolutions]
        self.wave_weight, self.power = float(wave_weight), float(power)
        for n_fft, _ in self.resolutions:
            self.register_buffer(f"win{n_fft}", torch.hann_window(n_fft), persistent=False)

    def _mag(self, x, n_fft, hop):
        S = torch.stft(x, n_fft, hop, window=getattr(self, f"win{n_fft}"), return_complex=True,
                       normalized=True)
        return torch.sqrt(S.real ** 2 + S.imag ** 2 + 1e-8)

    def forward(self, est: torch.Tensor, ref: torch.Tensor):
        wave = (est - ref).abs().mean()
        spec = est.new_zeros(())
        for n_fft, hop in self.resolutions:
            spec = spec + (self._mag(est, n_fft, hop) ** self.power
                           - self._mag(ref, n_fft, hop) ** self.power).abs().mean()
        spec = spec / len(self.resolutions)
        loss = self.wave_weight * wave + spec
        return loss, {"wave_l1": float(wave.detach()), "spec_l1": float(spec.detach())}
