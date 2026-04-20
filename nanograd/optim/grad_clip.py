"""Gradient clipping utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from nanograd.tensor import Tensor


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradients of parameters to have norm <= max_norm (in place).

    Returns the *total* gradient norm before clipping.
    """
    ps = [p for p in params if p.grad is not None]
    if not ps:
        return 0.0

    if norm_type == float("inf"):
        total = max(float(np.abs(p.grad).max()) for p in ps)
    else:
        sq = sum(float((p.grad ** norm_type).sum()) for p in ps)
        total = sq ** (1.0 / norm_type)

    clip = max_norm / (total + 1e-6)
    if clip < 1.0:
        for p in ps:
            p.grad *= clip
    return total


def clip_grad_value_(params: Iterable[Tensor], clip_value: float) -> None:
    """Clip gradients to [-clip_value, clip_value] element-wise."""
    for p in params:
        if p.grad is not None:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)
