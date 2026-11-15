import torch

from dwmss.losses import WaveMRSTFTLoss
from dwmss.model import MaskNet, count_parameters
from dwmss.stft import DualWindowSTFT


def test_param_count_independent_of_fft_size():
    counts = set()
    for N, M in [(1024, 1024), (256, 256), (2048, 256), (2048, 128)]:
        torch.manual_seed(0)
        counts.add(count_parameters(MaskNet()))
    assert len(counts) == 1


def test_chunked_equals_full_sequence():
    torch.manual_seed(0)
    s = DualWindowSTFT(2048, 256)
    model = MaskNet().eval()
    X = s.analysis(torch.randn(1, 30000))
    with torch.no_grad():
        full, _ = model(X, s.window_energy)
        parts, h = [], None
        for a in range(0, X.shape[1], 37):
            m, h = model(X[:, a:a + 37], s.window_energy, h)
            parts.append(m)
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-5)


def test_backward_is_finite_for_every_config():
    loss_fn = WaveMRSTFTLoss()
    for N, M in [(1024, 1024), (256, 256), (2048, 256)]:
        torch.manual_seed(0)
        s = DualWindowSTFT(N, M)
        model = MaskNet()
        mix, voc = torch.randn(2, 20000) * 0.1, torch.randn(2, 20000) * 0.05
        X = s.analysis(mix)
        mask, _ = model(X, s.window_energy)
        est = s.synthesis(mask * X, mix.shape[-1])
        assert est.shape == mix.shape
        loss, _ = loss_fn(est, voc)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
