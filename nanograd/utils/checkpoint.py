"""Checkpoint save/load via numpy .npz.

Stores parameter arrays keyed by their dotted name.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nanograd.nn.module import Module


def save(module: Module, path: str | Path) -> None:
    sd = module.state_dict()
    np.savez_compressed(path, **sd)


def load(module: Module, path: str | Path) -> None:
    with np.load(path) as f:
        state = {k: f[k] for k in f.files}
    module.load_state_dict(state)
