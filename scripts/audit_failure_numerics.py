"""Audit CPU/GPU inference differences without changing original evaluation artifacts."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from dwmss.data import STEMS, StemStore
from dwmss.metrics import si_sdr, usdr
from dwmss.model import MaskNet
from dwmss.separator import separate
from dwmss.stft import DualWindowSTFT


def main():
    spec = importlib.util.spec_from_file_location('failure_pipeline', REPO / 'scripts/analyze_failures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    track = 'PR - Happy Daze'
    output = REPO / 'results/failure_analysis_numerics'
    output.mkdir(exist_ok=True)
    rows = []
    defaults = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    for run in ('C_dual_seed0', 'C_dual_seed1'):
        torch.backends.cuda.matmul.allow_tf32 = defaults['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = defaults['cudnn_tf32']
        checkpoint = REPO / 'runs' / run / 'best.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = ck['config']
        store = StemStore(cfg['data']['musdb_root'], 'test', [track])
        vocal = store.read(track, 'vocals', 0, store.lengths[track])
        acc = sum(store.read(track, stem, 0, store.lengths[track]) for stem in STEMS[1:])
        mix = (vocal + acc).astype(np.float32)
        model = MaskNet(**cfg['model']).eval()
        model.load_state_dict(ck['model'])
        stft = DualWindowSTFT(cfg['stft']['n_analysis'], cfg['stft']['n_synthesis'])
        original = pd.read_csv(REPO / f'results/eval/{run}_test.csv').set_index('track').loc[track]
        start = time.monotonic()
        cpu = separate(model, stft, mix)
        rows.append({'run': run, 'method': 'original_separate_cpu', 'si_sdr': si_sdr(cpu, vocal),
                     'usdr': usdr(cpu, vocal), 'saved_si_sdr': original.si_sdr, 'saved_usdr': original.usdr,
                     'seconds': time.monotonic() - start})
        model, stft = model.to('cuda'), stft.to('cuda')
        previous = None
        for repeat in range(2):
            start = time.monotonic()
            gpu = separate(model, stft, mix)
            rows.append({'run': run, 'method': f'original_separate_cuda_repeat{repeat}',
                         'si_sdr': si_sdr(gpu, vocal), 'usdr': usdr(gpu, vocal),
                         'relative_l2_vs_cpu': float(np.linalg.norm(gpu.astype(np.float64) - cpu) / np.linalg.norm(cpu.astype(np.float64))),
                         'max_abs_vs_cpu': float(np.max(np.abs(gpu - cpu))),
                         'max_abs_vs_previous_gpu': float(np.max(np.abs(gpu - previous))) if previous is not None else None,
                         'seconds': time.monotonic() - start})
            previous = gpu
        start = time.monotonic()
        decomposed, _, _, relative, peak = module.decompose(model, stft, vocal, acc, 'cuda')
        rows.append({'run': run, 'method': 'failure_decompose_cuda', 'si_sdr': si_sdr(decomposed, vocal),
                     'usdr': usdr(decomposed, vocal), 'max_abs_vs_original_gpu': float(np.max(np.abs(decomposed - previous))),
                     'decomposition_relative_rms': relative, 'decomposition_max_abs': peak,
                     'seconds': time.monotonic() - start})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        start = time.monotonic()
        precise = separate(model, stft, mix)
        rows.append({'run': run, 'method': 'original_separate_cuda_tf32_disabled',
                     'si_sdr': si_sdr(precise, vocal), 'usdr': usdr(precise, vocal),
                     'relative_l2_vs_cpu': float(np.linalg.norm(precise.astype(np.float64) - cpu) / np.linalg.norm(cpu.astype(np.float64))),
                     'max_abs_vs_cpu': float(np.max(np.abs(precise - cpu))),
                     'seconds': time.monotonic() - start})
        print(json.dumps(rows[-5:], indent=2), flush=True)
        del model, stft
        torch.cuda.empty_cache()
    metadata = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                'default_matmul_tf32': defaults['matmul_tf32'], 'default_cudnn_tf32': defaults['cudnn_tf32'],
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'source_sha256': module.sha(Path(__file__)), 'rows': rows}
    module.write_json(output / 'audit.json', metadata)
    pd.DataFrame(rows).to_csv(output / 'audit.csv', index=False)


if __name__ == '__main__':
    main()
