"""Offline inference that is equivalent to frame-by-frame streaming (chunked GRU state)."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def separate(model, stft, mix: np.ndarray, chunk_frames: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.as_tensor(mix, dtype=torch.float32, device=device)[None]
    X = stft.analysis(x)
    masks, h = [], None
    for s in range(0, X.shape[1], chunk_frames):
        m, h = model(X[:, s:s + chunk_frames], stft.window_energy, h)
        masks.append(m)
    est = stft.synthesis(torch.cat(masks, dim=1) * X, x.shape[-1])
    return est[0].float().cpu().numpy()
