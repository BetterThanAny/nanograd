"""Weight initialization helpers.

All functions mutate the given Tensor's ``data`` in place.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from nanograd.tensor import Tensor


def _compute_fans(shape) -> tuple[int, int]:
    """Compute fan_in / fan_out from a weight shape.

    - (in, out) for Linear: fan_in=in, fan_out=out
    - (out, in, kh, kw) for Conv2d: fan_in=in*kh*kw, fan_out=out*kh*kw
    """
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 4:
        # conv layout (out, in, kh, kw)
        receptive = shape[2] * shape[3]
        return shape[1] * receptive, shape[0] * receptive
    # fallback
    return int(np.prod(shape)), int(np.prod(shape))


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, nonlinearity: str = "relu") -> None:
    fan_in, _ = _compute_fans(tensor.shape)
    gain = _gain(nonlinearity, a)
    bound = gain * math.sqrt(3.0 / fan_in)
    rng = np.random.default_rng()
    tensor.data[:] = rng.uniform(-bound, bound, size=tensor.shape).astype(tensor.dtype)


def kaiming_normal_(tensor: Tensor, a: float = 0.0, nonlinearity: str = "relu") -> None:
    fan_in, _ = _compute_fans(tensor.shape)
    gain = _gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    rng = np.random.default_rng()
    tensor.data[:] = (rng.standard_normal(size=tensor.shape) * std).astype(tensor.dtype)


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> None:
    fan_in, fan_out = _compute_fans(tensor.shape)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    rng = np.random.default_rng()
    tensor.data[:] = rng.uniform(-bound, bound, size=tensor.shape).astype(tensor.dtype)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> None:
    fan_in, fan_out = _compute_fans(tensor.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    rng = np.random.default_rng()
    tensor.data[:] = (rng.standard_normal(size=tensor.shape) * std).astype(tensor.dtype)


def zeros_(tensor: Tensor) -> None:
    tensor.data[:] = 0.0


def ones_(tensor: Tensor) -> None:
    tensor.data[:] = 1.0


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    rng = np.random.default_rng()
    tensor.data[:] = (rng.standard_normal(size=tensor.shape) * std + mean).astype(tensor.dtype)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> None:
    rng = np.random.default_rng()
    tensor.data[:] = rng.uniform(a, b, size=tensor.shape).astype(tensor.dtype)


def _gain(nonlinearity: str, a: float = 0.0) -> float:
    table = {
        "linear": 1.0,
        "conv2d": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0 / (1 + a * a)),
    }
    if nonlinearity not in table:
        raise ValueError(f"unknown nonlinearity {nonlinearity!r}")
    return table[nonlinearity]
