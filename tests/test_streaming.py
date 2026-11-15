import numpy as np
import pytest
import torch

from dwmss.export import FrameStep, OnnxStep, export_onnx
from dwmss.model import MaskNet
from dwmss.separator import separate
from dwmss.stft import DualWindowSTFT

CONFIGS = [(1024, 1024), (256, 256), (2048, 256), (2048, 128)]


class Ones(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X, energy, h=None):
        return torch.ones(X.shape, dtype=torch.float32), h


@pytest.mark.parametrize("N,M", CONFIGS)
def test_impulse_alignment(N, M):
    from dwmss.streaming import StreamingSeparator
    s = DualWindowSTFT(N, M)
    st = StreamingSeparator(Ones(), s)
    x = np.zeros(8000, dtype=np.float32)
    x[3001] = 1.0
    y = st.run(x)
    assert int(np.argmax(np.abs(y))) == 3001 + st.output_delay
    assert y[3001 + st.output_delay] == pytest.approx(1.0, abs=1e-5)
    assert st.output_delay + st.R - 1 == M - 1  # worst-case algorithmic latency


@pytest.mark.parametrize("N,M", CONFIGS)
def test_streaming_equals_offline(N, M):
    from dwmss.streaming import StreamingSeparator
    torch.manual_seed(0)
    s = DualWindowSTFT(N, M)
    model = MaskNet().eval()
    x = (np.random.default_rng(0).standard_normal(6000) * 0.1).astype(np.float32)
    offline = separate(model, s, x)
    st = StreamingSeparator(model, s)
    y = st.run(np.concatenate([x, np.zeros(st.output_delay, np.float32)]))
    d = st.output_delay
    assert np.max(np.abs(y[d:d + len(x)] - offline)) < 1e-4


def test_frame_step_and_onnx_match_model(tmp_path):
    torch.manual_seed(0)
    s = DualWindowSTFT(2048, 256)
    model = MaskNet().eval()
    X = s.analysis(torch.randn(1, 5000) * 0.1)[:, :40]
    with torch.no_grad():
        ref, _ = model(X, s.window_energy)
        step = FrameStep(model, s.n_bins)
        h = torch.zeros(1, 1, model.gru.hidden_size)
        logpow = torch.log((X.real ** 2 + X.imag ** 2) / s.window_energy + 1e-10)
        outs = []
        for t in range(X.shape[1]):
            m, h = step(logpow[:, t], h)
            outs.append(m)
    assert torch.allclose(torch.stack(outs, 1), ref, atol=1e-5)
    path = str(tmp_path / "step.onnx")
    export_onnx(model, s.n_bins, path)
    ort_step, hh, outs = OnnxStep(path, model.gru.hidden_size), None, []
    for t in range(X.shape[1]):
        m, hh = ort_step(logpow[0, t].numpy(), hh)
        outs.append(m)
    assert np.max(np.abs(np.stack(outs) - ref[0].numpy())) < 1e-4
