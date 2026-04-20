"""Elementwise op fusion — share one output buffer across a chain.

Forward pass of a chain `f_n ∘ ... ∘ f_1` normally allocates (n-1) intermediate
arrays. `FusedChain` allocates a single scratch buffer and reuses it via numpy's
``out=`` parameter, eliminating per-op allocations.

Backward pass recomputes intermediates (since buffer was overwritten) and
composes per-op VJPs.

Supported ops (name, optional scalar arg):
  unary: "relu", "exp", "log", "neg", "abs", "sqrt", "tanh"
  binary-with-scalar: ("add", c), ("sub", c), ("mul", c), ("pow", c)
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from nanograd.function import Function


Op = Union[str, Tuple[str, float]]


def _apply_forward(src: np.ndarray, out: np.ndarray, op: Op) -> None:
    if isinstance(op, tuple):
        name, c = op
        if name == "add":
            np.add(src, c, out=out)
        elif name == "sub":
            np.subtract(src, c, out=out)
        elif name == "mul":
            np.multiply(src, c, out=out)
        elif name == "pow":
            np.power(src, c, out=out)
        else:
            raise ValueError(f"unknown scalar op {name}")
        return

    if op == "relu":
        np.maximum(src, 0, out=out)
    elif op == "exp":
        np.exp(src, out=out)
    elif op == "log":
        np.log(src, out=out)
    elif op == "neg":
        np.negative(src, out=out)
    elif op == "abs":
        np.abs(src, out=out)
    elif op == "sqrt":
        np.sqrt(src, out=out)
    elif op == "tanh":
        np.tanh(src, out=out)
    else:
        raise ValueError(f"unknown unary op {op}")


def _backward_op(src: np.ndarray, out: np.ndarray, g: np.ndarray, op: Op) -> np.ndarray:
    if isinstance(op, tuple):
        name, c = op
        if name == "add":
            return g
        if name == "sub":
            return g
        if name == "mul":
            return g * c
        if name == "pow":
            return g * c * np.power(src, c - 1)
        raise ValueError(name)

    if op == "relu":
        return g * (src > 0).astype(g.dtype)
    if op == "exp":
        return g * out
    if op == "log":
        return g / src
    if op == "neg":
        return -g
    if op == "abs":
        return g * np.sign(src)
    if op == "sqrt":
        return 0.5 * g / out
    if op == "tanh":
        return g * (1 - out * out)
    raise ValueError(op)


class FusedChain(Function):
    """Apply a list of elementwise ops to a tensor in a single buffer."""

    def forward(self, x, *, ops: List[Op]):
        if not ops:
            return x.copy()
        buf = np.empty_like(x)
        _apply_forward(x, buf, ops[0])
        for op in ops[1:]:
            _apply_forward(buf, buf, op)
        self.ops = ops
        # for backward we need all intermediates; save x, and will recompute
        self.save_for_backward(x)
        return buf

    def backward(self, g):
        (x,) = self.saved
        ops = self.ops
        # re-run forward storing each intermediate
        inters = [x]
        cur = x
        for op in ops:
            nxt = np.empty_like(x)
            _apply_forward(cur, nxt, op)
            inters.append(nxt)
            cur = nxt
        # backward: iterate ops in reverse, each receives src, out, g
        dg = g
        for i in range(len(ops) - 1, -1, -1):
            dg = _backward_op(inters[i], inters[i + 1], dg, ops[i])
        return (dg,)


def fused(x, ops: List[Op]):
    from nanograd.tensor import Tensor

    return FusedChain.apply(x, ops=ops)
