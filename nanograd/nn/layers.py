from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from nanograd.function import Function
from nanograd.nn import functional as F
from nanograd.nn.module import Module, Parameter
from nanograd.tensor import Tensor


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: Optional[int] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(seed)
        # Kaiming uniform (matches PyTorch default for linear)
        bound = 1 / math.sqrt(in_features)
        w = rng.uniform(-bound, bound, size=(in_features, out_features)).astype(np.float32)
        self.weight = Parameter(w)
        if bias:
            b = rng.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
            self.bias = Parameter(b)
        else:
            self.bias = None
            self._modules.pop("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, f"layer_{i}", m)
        self._layer_names = [f"layer_{i}" for i in range(len(modules))]

    def forward(self, x: Tensor) -> Tensor:
        for name in self._layer_names:
            x = getattr(self, name)(x)
        return x

    def __iter__(self):
        for name in self._layer_names:
            yield getattr(self, name)

    def __len__(self) -> int:
        return len(self._layer_names)


# ---------------------------------------------------------------------------
# activation modules (wrap functional)
# ---------------------------------------------------------------------------


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, axis=self.axis)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


class _Dropout(Function):
    def forward(self, a, *, p, mask):
        self.save_for_backward(mask)
        self.scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
        return a * mask * self.scale

    def backward(self, g):
        (mask,) = self.saved
        return (g * mask * self.scale,)


class Dropout(Module):
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout p must be in [0,1), got {p}")
        self.p = p
        self._rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (self._rng.random(x.shape) >= self.p).astype(np.float32)
        return _Dropout.apply(x, p=self.p, mask=mask)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


class _LayerNormFn(Function):
    def forward(self, x, gamma, beta, *, eps):
        # normalize over last dim(s) matching gamma.shape
        axes = tuple(range(x.ndim - gamma.ndim, x.ndim))
        mean = x.mean(axis=axes, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xhat = (x - mean) * inv
        out = xhat * gamma + beta
        self.save_for_backward(xhat, gamma, inv, axes)
        return out.astype(x.dtype)

    def backward(self, g):
        xhat, gamma, inv, axes = self.saved
        N = 1
        for a in axes:
            N *= xhat.shape[a]
        # dL/dxhat = g * gamma
        dxhat = g * gamma
        # LN backward formula
        dx = (1.0 / N) * inv * (
            N * dxhat
            - dxhat.sum(axis=axes, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=axes, keepdims=True)
        )
        # grad gamma, beta: sum over all but feature dims
        reduce_axes = tuple(i for i in range(g.ndim) if i not in axes)
        dgamma = (g * xhat).sum(axis=reduce_axes) if reduce_axes else (g * xhat)
        dbeta = g.sum(axis=reduce_axes) if reduce_axes else g
        return dx, dgamma, dbeta


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
        self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return _LayerNormFn.apply(x, self.weight, self.bias, eps=self.eps)
