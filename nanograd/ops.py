"""Elementwise + basic ops for M1.

Each op is a subclass of Function with forward/backward.
Registered as Tensor methods at import time.
"""
from __future__ import annotations

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
        # db uses log; guard against non-positive a by masking
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


# ---------------------------------------------------------------------------
# tensor methods registration
# ---------------------------------------------------------------------------


def _wrap(other) -> Tensor:
    if isinstance(other, Tensor):
        return other
    return Tensor(_as_array(other))


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

Tensor.exp = lambda self: Exp.apply(self)
Tensor.log = lambda self: Log.apply(self)
