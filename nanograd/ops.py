"""Core tensor ops.

Layout:
  - elementwise binary: Add, Sub, Mul, Div, Pow
  - elementwise unary:  Neg, Exp, Log, Sqrt, Abs
  - reductions:         Sum, Mean, Max
  - linalg:             MatMul
  - shape:              Reshape, Transpose, Expand
  - indexing (minimal): Getitem

Each op is a Function subclass; registered on Tensor at import time.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from nanograd.function import Function
from nanograd.tensor import Tensor, _as_array


# ---------------------------------------------------------------------------
# elementwise binary
# ---------------------------------------------------------------------------


class Add(Function):
    def forward(self, a, b):
        return a + b

    def backward(self, g):
        return g, g


class Sub(Function):
    def forward(self, a, b):
        return a - b

    def backward(self, g):
        return g, -g


class Mul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a * b

    def backward(self, g):
        a, b = self.saved
        return g * b, g * a


class Div(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a / b

    def backward(self, g):
        a, b = self.saved
        return g / b, -g * a / (b * b)


class Pow(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return np.power(a, b)

    def backward(self, g):
        a, b = self.saved
        da = g * b * np.power(a, b - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_a = np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), 0.0)
        db = g * np.power(a, b) * log_a
        return da, db


# ---------------------------------------------------------------------------
# elementwise unary
# ---------------------------------------------------------------------------


class Neg(Function):
    def forward(self, a):
        return -a

    def backward(self, g):
        return (-g,)


class Exp(Function):
    def forward(self, a):
        out = np.exp(a)
        self.save_for_backward(out)
        return out

    def backward(self, g):
        (out,) = self.saved
        return (g * out,)


class Log(Function):
    def forward(self, a):
        self.save_for_backward(a)
        return np.log(a)

    def backward(self, g):
        (a,) = self.saved
        return (g / a,)


class Sqrt(Function):
    def forward(self, a):
        out = np.sqrt(a)
        self.save_for_backward(out)
        return out

    def backward(self, g):
        (out,) = self.saved
        return (g * 0.5 / out,)


class Abs(Function):
    def forward(self, a):
        self.save_for_backward(a)
        return np.abs(a)

    def backward(self, g):
        (a,) = self.saved
        return (g * np.sign(a),)


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


class Sum(Function):
    def forward(self, a, *, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.in_shape = a.shape
        return a.sum(axis=axis, keepdims=keepdims)

    def backward(self, g):
        out = _reduce_grad(g, self.in_shape, self.axis, self.keepdims)
        return (out,)


class Mean(Function):
    def forward(self, a, *, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.in_shape = a.shape
        self.n = _reduced_count(a.shape, axis)
        return a.mean(axis=axis, keepdims=keepdims)

    def backward(self, g):
        out = _reduce_grad(g, self.in_shape, self.axis, self.keepdims) / self.n
        return (out,)


class Max(Function):
    def forward(self, a, *, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.in_shape = a.shape
        out = a.max(axis=axis, keepdims=True)
        # mask of max elements (ties split uniformly)
        mask = (a == out).astype(a.dtype)
        mask = mask / mask.sum(axis=axis, keepdims=True)
        self.save_for_backward(mask)
        if not keepdims:
            out = out.squeeze(axis=axis) if axis is not None else out.reshape(())
        return out

    def backward(self, g):
        (mask,) = self.saved
        grad = _reduce_grad(g, self.in_shape, self.axis, self.keepdims)
        return (grad * mask,)


def _reduced_count(shape, axis) -> int:
    if axis is None:
        n = 1
        for s in shape:
            n *= s
        return n
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    n = 1
    for a in axes:
        n *= shape[a]
    return n


def _reduce_grad(g: np.ndarray, in_shape: Tuple[int, ...], axis, keepdims: bool) -> np.ndarray:
    """Expand grad from reduced shape back to input shape."""
    if axis is None:
        return np.broadcast_to(g, in_shape).copy()
    if not keepdims:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        for a in sorted([ax % len(in_shape) for ax in axes]):
            g = np.expand_dims(g, a)
    return np.broadcast_to(g, in_shape).copy()


# ---------------------------------------------------------------------------
# linalg
# ---------------------------------------------------------------------------


class MatMul(Function):
    """Supports 2D x 2D and batched (..., M, K) x (..., K, N)."""

    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a @ b

    def backward(self, g):
        a, b = self.saved
        # swap last two dims
        da = g @ _swap_last_two(b)
        db = _swap_last_two(a) @ g
        return da, db


def _swap_last_two(x: np.ndarray) -> np.ndarray:
    if x.ndim < 2:
        return x
    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return np.transpose(x, axes)


# ---------------------------------------------------------------------------
# shape ops
# ---------------------------------------------------------------------------


class Reshape(Function):
    def forward(self, a, *, shape):
        self.in_shape = a.shape
        return a.reshape(shape)

    def backward(self, g):
        return (g.reshape(self.in_shape),)


class Transpose(Function):
    def forward(self, a, *, axes=None):
        self.axes = axes
        return np.transpose(a, axes)

    def backward(self, g):
        if self.axes is None:
            return (np.transpose(g),)
        # invert permutation
        inv = [0] * len(self.axes)
        for i, a in enumerate(self.axes):
            inv[a] = i
        return (np.transpose(g, inv),)


class Expand(Function):
    """Broadcast a tensor to a larger shape. grad reduces via _unbroadcast (done by backward runner)."""

    def forward(self, a, *, shape):
        self.in_shape = a.shape
        return np.broadcast_to(a, shape).copy()

    def backward(self, g):
        # grad will be auto-unbroadcast to parent shape by the engine
        return (g,)


# ---------------------------------------------------------------------------
# indexing (slice / integer)
# ---------------------------------------------------------------------------


class Getitem(Function):
    def forward(self, a, *, idx):
        self.in_shape = a.shape
        self.idx = idx
        return a[idx]

    def backward(self, g):
        out = np.zeros(self.in_shape, dtype=g.dtype)
        np.add.at(out, self.idx, g)
        return (out,)


def _unwrap_idx(idx):
    """Allow Tensor-as-index by pulling out .data (and coercing float → int64)."""
    if isinstance(idx, Tensor):
        d = idx.data
        return d.astype(np.int64) if np.issubdtype(d.dtype, np.floating) else d
    if isinstance(idx, tuple):
        return tuple(_unwrap_idx(i) for i in idx)
    return idx


# ---------------------------------------------------------------------------
# concat / stack / pad
# ---------------------------------------------------------------------------


class Concat(Function):
    def forward(self, *arrs, axis):
        self.axis = axis
        self.sizes = [a.shape[axis] for a in arrs]
        return np.concatenate(arrs, axis=axis)

    def backward(self, g):
        splits = np.cumsum(self.sizes)[:-1]
        return tuple(np.split(g, splits, axis=self.axis))


class Stack(Function):
    def forward(self, *arrs, axis):
        self.axis = axis
        self.n = len(arrs)
        return np.stack(arrs, axis=axis)

    def backward(self, g):
        return tuple(np.take(g, i, axis=self.axis) for i in range(self.n))


class Pad(Function):
    """Zero-padding. pad_widths: list of (before, after) tuples, one per axis."""

    def forward(self, a, *, pad_widths):
        self.pad_widths = pad_widths
        return np.pad(a, pad_widths)

    def backward(self, g):
        slices = tuple(slice(p0, g.shape[i] - p1) for i, (p0, p1) in enumerate(self.pad_widths))
        return (g[slices].copy(),)


# ---------------------------------------------------------------------------
# registration on Tensor
# ---------------------------------------------------------------------------


def _wrap(other) -> Tensor:
    if isinstance(other, Tensor):
        return other
    return Tensor(_as_array(other))


# binary
Tensor.__add__ = lambda self, other: Add.apply(self, _wrap(other))
Tensor.__radd__ = lambda self, other: Add.apply(_wrap(other), self)
Tensor.__sub__ = lambda self, other: Sub.apply(self, _wrap(other))
Tensor.__rsub__ = lambda self, other: Sub.apply(_wrap(other), self)
Tensor.__mul__ = lambda self, other: Mul.apply(self, _wrap(other))
Tensor.__rmul__ = lambda self, other: Mul.apply(_wrap(other), self)
Tensor.__truediv__ = lambda self, other: Div.apply(self, _wrap(other))
Tensor.__rtruediv__ = lambda self, other: Div.apply(_wrap(other), self)
Tensor.__neg__ = lambda self: Neg.apply(self)
Tensor.__pow__ = lambda self, other: Pow.apply(self, _wrap(other))
Tensor.__matmul__ = lambda self, other: MatMul.apply(self, _wrap(other))

# unary
Tensor.exp = lambda self: Exp.apply(self)
Tensor.log = lambda self: Log.apply(self)
Tensor.sqrt = lambda self: Sqrt.apply(self)
Tensor.abs = lambda self: Abs.apply(self)

# reductions
Tensor.sum = lambda self, axis=None, keepdims=False: Sum.apply(self, axis=axis, keepdims=keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: Mean.apply(self, axis=axis, keepdims=keepdims)
Tensor.max = lambda self, axis=None, keepdims=False: Max.apply(self, axis=axis, keepdims=keepdims)

# shape
Tensor.reshape = lambda self, *shape: Reshape.apply(
    self, shape=shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
)
Tensor.transpose = lambda self, *axes: Transpose.apply(self, axes=axes if axes else None)
Tensor.T = property(lambda self: Transpose.apply(self, axes=None))
Tensor.expand = lambda self, *shape: Expand.apply(
    self, shape=shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
)
Tensor.matmul = lambda self, other: MatMul.apply(self, _wrap(other))

# indexing
Tensor.__getitem__ = lambda self, idx: Getitem.apply(self, idx=_unwrap_idx(idx))


# module-level convenience functions
def cat(tensors, axis: int = 0):
    if not tensors:
        raise ValueError("cat requires at least one tensor")
    return Concat.apply(*tensors, axis=axis)


def stack(tensors, axis: int = 0):
    if not tensors:
        raise ValueError("stack requires at least one tensor")
    return Stack.apply(*tensors, axis=axis)


def pad(tensor, pad_widths):
    return Pad.apply(tensor, pad_widths=pad_widths)
