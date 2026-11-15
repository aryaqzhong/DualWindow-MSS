"""YAML configs with single-parent inheritance (``inherit: base.yaml``)."""
from __future__ import annotations

import copy
import os

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), parent)), cfg)
    return cfg
