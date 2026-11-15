"""Frame-by-frame streaming separation (same arithmetic as the offline path).

Each ``process`` call consumes R = M/2 new input samples and returns R output samples. The output
equals the offline output delayed by ``M - R`` samples; together with the R-sample input block this
gives the algorithmic latency of M - 1 samples in the worst case (see tests/test_streaming.py).
"""
from __future__ import annotations

import numpy as np
import torch


class StreamingSeparator:
    def __init__(self, model, stft, step=None):
        """model: MaskNet (eager PyTorch). step: optional callable(logpow (F,), h) -> (mask (F,), h),
        e.g. an ONNX Runtime session wrapper; when given, the PyTorch model is not used per frame."""
        self.N, self.M, self.R = stft.N, stft.M, stft.R
        self.wa = stft.wa.detach().cpu().numpy().astype(np.float32)
        self.ws = stft.ws_tail.detach().cpu().numpy().astype(np.float32)
        self.energy = float(stft.window_energy)
        self.model = model.eval() if model is not None else None
        self.step = step
        self.reset()

    @property
    def output_delay(self) -> int:
        return self.M - self.R

    def reset(self):
        self.buf = np.zeros(self.N, dtype=np.float32)
        self.ola = np.zeros(self.R, dtype=np.float32)
        self.h = None
        # The offline path starts with an all-zero frame (left zero-padding); step the recurrent state
        # through that frame too so streaming and offline outputs are identical. Its output is zero.
        if self.model is not None or self.step is not None:
            self._mask(np.fft.rfft(self.buf * self.wa))

    def _mask(self, X: np.ndarray) -> np.ndarray:
        if self.step is not None:
            logpow = np.log((X.real ** 2 + X.imag ** 2) / self.energy + 1e-10).astype(np.float32)
            mask, self.h = self.step(logpow, self.h)
            return mask
        with torch.inference_mode():
            m, self.h = self.model(torch.from_numpy(X.astype(np.complex64))[None, None], self.energy, self.h)
        return m[0, 0].numpy()

    def process(self, x: np.ndarray) -> np.ndarray:
        self.buf[:-self.R] = self.buf[self.R:]
        self.buf[-self.R:] = x
        X = np.fft.rfft(self.buf * self.wa)
        y = np.fft.irfft(self._mask(X) * X, n=self.N)[self.N - self.M:] * self.ws
        out = self.ola + y[:self.R]
        self.ola = y[self.R:].astype(np.float32)
        return out.astype(np.float32)

    def run(self, x: np.ndarray) -> np.ndarray:
        """Stream a whole signal (zero-padded to whole hops); returns the delayed output."""
        n = int(np.ceil(len(x) / self.R)) * self.R
        xp = np.zeros(n, dtype=np.float32)
        xp[:len(x)] = x
        return np.concatenate([self.process(xp[i:i + self.R]) for i in range(0, n, self.R)])
