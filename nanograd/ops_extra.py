"""Additional ops: flip, roll, gather, scatter_add. Registered on Tensor at import."""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from nanograd.function import Function
from nanograd.tensor import Tensor


class Flip(Function):
    def forward(self, a, *, axis):
        self.axis = axis
        return np.flip(a, axis=axis).copy()

    def backward(self, g):
        return (np.flip(g, axis=self.axis).copy(),)


class Roll(Function):
    def forward(self, a, *, shift, axis):
        self.shift = shift
        self.axis = axis
        return np.roll(a, shift=shift, axis=axis).copy()

    def backward(self, g):
        # reverse: roll by negative shift
        neg = tuple(-s for s in self.shift) if isinstance(self.shift, tuple) else -self.shift
        return (np.roll(g, shift=neg, axis=self.axis).copy(),)


class Gather(Function):
    """out[i0,...,iN] = a[i0,..., idx[i0,...,iN], ..., iN]   (along ``axis``).

    Shapes: ``a`` and ``idx`` must agree everywhere except along ``axis``.
    """

    def forward(self, a, *, index, axis):
        self.in_shape = a.shape
        self.axis = axis
        self.index = index
        return np.take_along_axis(a, index, axis=axis)

    def backward(self, g):
        grad = np.zeros(self.in_shape, dtype=g.dtype)
        # equivalent of torch.scatter_add_: grad.take_along_axis(index, axis) += g
        np.put_along_axis(grad, self.index, 0, axis=self.axis)  # no-op assign for type
        # np.add.at along arbitrary axis via building broadcasted coords
        idx_grid = np.indices(self.index.shape)
        coords = list(idx_grid)
        coords[self.axis] = self.index
        np.add.at(grad, tuple(coords), g)
        return (grad,)


class ScatterAdd(Function):
    """Accumulate ``src`` into ``base`` at positions ``index`` along ``axis``.

    Like torch ``base.scatter_add(axis, index, src)``. Returns a new tensor
    (does not mutate base).
    """

    def forward(self, base, src, *, index, axis):
        self.base_shape = base.shape
        self.src_shape = src.shape
        self.index = index
        self.axis = axis
        out = base.copy()
        idx_grid = np.indices(index.shape)
        coords = list(idx_grid)
        coords[axis] = index
        np.add.at(out, tuple(coords), src)
        return out

    def backward(self, g):
        # grad w.r.t. base: g itself (scatter_add is base + contributions)
        grad_base = g
        # grad w.r.t. src: gather g at index
        grad_src = np.take_along_axis(g, self.index, axis=self.axis)
        return grad_base, grad_src


# -------- registration --------


def flip(tensor, axis):
    return Flip.apply(tensor, axis=axis)


def roll(tensor, shift, axis=None):
    return Roll.apply(tensor, shift=shift, axis=axis)


def gather(tensor, index, axis: int):
    if isinstance(index, Tensor):
        index = index.data
    index = np.asarray(index, dtype=np.int64)
    return Gather.apply(tensor, index=index, axis=axis)


def scatter_add(base, index, src, axis: int):
    if isinstance(index, Tensor):
        index = index.data
    index = np.asarray(index, dtype=np.int64)
    return ScatterAdd.apply(base, src, index=index, axis=axis)


Tensor.flip = lambda self, axis: flip(self, axis)
Tensor.roll = lambda self, shift, axis=None: roll(self, shift, axis)
Tensor.gather = lambda self, index, axis: gather(self, index, axis)
Tensor.scatter_add = lambda self, index, src, axis: scatter_add(self, index, src, axis)


# -------- std / var as pure compositions --------


def _axis_size(shape, axis):
    if axis is None:
        return int(np.prod(shape))
    if isinstance(axis, (tuple, list)):
        return int(np.prod([shape[a] for a in axis]))
    return shape[axis]


def _var(self, axis=None, keepdims: bool = False, unbiased: bool = False):
    m = self.mean(axis=axis, keepdims=True)
    diff = self - m
    sq_sum = (diff * diff).sum(axis=axis, keepdims=keepdims)
    n = _axis_size(self.shape, axis)
    denom = n - 1 if unbiased else n
    assert denom > 0, "var: denominator is zero"
    return sq_sum * (1.0 / denom)


def _std(self, axis=None, keepdims: bool = False, unbiased: bool = False):
    return _var(self, axis=axis, keepdims=keepdims, unbiased=unbiased).sqrt()


Tensor.var = _var
Tensor.std = _std
