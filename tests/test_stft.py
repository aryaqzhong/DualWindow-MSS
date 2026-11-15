import math

import numpy as np
import pytest
import torch

from dwmss.stft import DualWindowSTFT, make_windows

CONFIGS = [(1024, 1024), (256, 256), (2048, 256), (2048, 128), (4096, 256)]


def loop_reference(x, N, M, mask):
    """Plain per-frame loop implementation (independent of the vectorized overlap-add)."""
    wa, ws = make_windows(N, M)
    R = M // 2
    xp = np.pad(x, (N, N))
    T = (len(x) + N) // R + 1
    out = np.zeros(len(xp) + N)
    for t in range(T):
        spec = np.fft.rfft(xp[t * R:t * R + N] * wa) * mask[t]
        y = np.fft.irfft(spec, n=N) * ws
        out[t * R + N - M:t * R + N] += y[N - M:]
    return out[N:N + len(x)]


@pytest.mark.parametrize("N,M", CONFIGS)
def test_perfect_reconstruction(N, M):
    torch.manual_seed(0)
    x = torch.randn(2, 44100 + 123)
    s = DualWindowSTFT(N, M)
    y = s.synthesis(s.analysis(x), x.shape[-1])
    snr = 10 * math.log10(float(x.pow(2).sum() / (y - x).pow(2).sum()))
    assert snr > 100, snr


@pytest.mark.parametrize("N,M", CONFIGS)
def test_frame_count(N, M):
    s = DualWindowSTFT(N, M)
    for L in (1000, 4410, 12345):
        assert s.analysis(torch.zeros(1, L)).shape[1] == s.n_frames(L)


@pytest.mark.parametrize("N,M", CONFIGS)
def test_matches_loop_reference_with_mask(N, M):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(6000)
    s = DualWindowSTFT(N, M)
    wa, ws = make_windows(N, M)  # exact float64 windows (the module stores float32 copies)
    s.wa, s.ws_tail = torch.tensor(wa), torch.tensor(ws[N - M:])
    X = s.analysis(torch.tensor(x)[None])
    mask = rng.uniform(0, 1, size=X.shape[1:])
    y = s.synthesis(X * torch.tensor(mask)[None], len(x))[0].numpy()
    ref = loop_reference(x, N, M, mask)
    assert np.max(np.abs(y - ref)) < 1e-9


@pytest.mark.parametrize("N,M", [(1024, 1024), (256, 256), (2048, 256), (2048, 128)])
def test_prefix_causality(N, M):
    """Output samples must not depend on input more than M - 1 samples in the future."""
    torch.manual_seed(0)
    s = DualWindowSTFT(N, M)
    L, p0_base = 30000, 15000
    base = torch.randn(1, L)

    def process(x):
        X = s.analysis(x)
        mag = X.abs()
        m = mag / (mag + mag.median(dim=-1, keepdim=True).values + 1e-8)  # frame-local nonlinearity
        return s.synthesis(m * X, L)[0]

    worst = 0
    for off in range(0, s.R, max(1, s.R // 8)):
        p0 = p0_base + off
        x2 = base.clone()
        x2[0, p0:] = torch.randn(L - p0)
        diff = torch.nonzero((process(base) - process(x2)).abs() > 1e-6)
        first = int(diff[0]) if len(diff) else L
        worst = max(worst, p0 - first)
    assert worst <= M - 1, (worst, M)
