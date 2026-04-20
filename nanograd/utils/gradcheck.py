"""Numerical gradient check utility."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from nanograd.tensor import Tensor


def numerical_grad(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-3,
) -> list[np.ndarray]:
    """Compute numerical gradients of scalar output `fn(*inputs).sum()` wrt each input."""
    grads: list[np.ndarray] = []
    for idx, t in enumerate(inputs):
        g = np.zeros_like(t.data)
        flat = t.data.reshape(-1)
        g_flat = g.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            plus = _scalar(fn(*inputs))
            flat[i] = orig - eps
            minus = _scalar(fn(*inputs))
            flat[i] = orig
            g_flat[i] = (plus - minus) / (2 * eps)
        grads.append(g)
    return grads


def _scalar(t: Tensor) -> float:
    return float(t.data.sum())


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-3,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Compare analytic vs numerical gradients. Returns True if all close.

    All inputs must have requires_grad=True. fn should return a Tensor; we
    reduce via .sum() to get a scalar before calling backward.
    """
    for t in inputs:
        t.zero_grad()
        assert t.requires_grad, "inputs must have requires_grad=True"

    out = fn(*inputs)
    if out.data.size != 1:
        # reduce to scalar via sum of all entries
        g = np.ones_like(out.data)
        out.backward(g)
    else:
        out.backward()

    analytic = [t.grad if t.grad is not None else np.zeros_like(t.data) for t in inputs]
    numeric = numerical_grad(fn, inputs, eps=eps)

    for i, (a, n) in enumerate(zip(analytic, numeric)):
        if not np.allclose(a, n, atol=atol, rtol=rtol):
            diff = np.max(np.abs(a - n))
            raise AssertionError(
                f"grad mismatch on input {i}: max abs diff = {diff:g}\n"
                f"analytic=\n{a}\nnumeric=\n{n}"
            )
    return True
