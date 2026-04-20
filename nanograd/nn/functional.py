"""Stable activation + loss functions.

Implemented as Function subclasses for numerical stability and efficiency,
rather than composed ops. All accept and return `Tensor`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from nanograd.function import Function
from nanograd.tensor import Tensor


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------


class ReLU(Function):
    def forward(self, a):
        out = np.maximum(a, 0)
        self.save_for_backward(a)
        return out

    def backward(self, g):
        (a,) = self.saved
        return (g * (a > 0).astype(g.dtype),)


class Sigmoid(Function):
    def forward(self, a):
        # stable sigmoid
        out = np.where(
            a >= 0,
            1.0 / (1.0 + np.exp(-a)),
            np.exp(a) / (1.0 + np.exp(a)),
        ).astype(a.dtype)
        self.save_for_backward(out)
        return out

    def backward(self, g):
        (out,) = self.saved
        return (g * out * (1 - out),)


class Tanh(Function):
    def forward(self, a):
        out = np.tanh(a).astype(a.dtype)
        self.save_for_backward(out)
        return out

    def backward(self, g):
        (out,) = self.saved
        return (g * (1 - out * out),)


class LeakyReLU(Function):
    def forward(self, a, *, negative_slope=0.01):
        self.negative_slope = negative_slope
        self.save_for_backward(a)
        return np.where(a > 0, a, negative_slope * a).astype(a.dtype)

    def backward(self, g):
        (a,) = self.saved
        return (g * np.where(a > 0, 1.0, self.negative_slope).astype(g.dtype),)


class GELU(Function):
    """Approximate GELU (tanh-based, matches PyTorch's approximate='tanh')."""

    _C = np.float32(np.sqrt(2.0 / np.pi))

    def forward(self, a):
        k = self._C * (a + 0.044715 * a ** 3)
        t = np.tanh(k)
        out = 0.5 * a * (1 + t)
        self.save_for_backward(a, t)
        return out.astype(a.dtype)

    def backward(self, g):
        a, t = self.saved
        dk_da = self._C * (1 + 3 * 0.044715 * a * a)
        dt_da = (1 - t * t) * dk_da
        grad = 0.5 * (1 + t) + 0.5 * a * dt_da
        return (g * grad.astype(g.dtype),)


class Softmax(Function):
    def forward(self, a, *, axis=-1):
        self.axis = axis
        shifted = a - a.max(axis=axis, keepdims=True)
        exp = np.exp(shifted)
        out = exp / exp.sum(axis=axis, keepdims=True)
        self.save_for_backward(out)
        return out.astype(a.dtype)

    def backward(self, g):
        (out,) = self.saved
        # dL/dx = out * (g - (g*out).sum(axis, keepdims=True))
        sum_gy = (g * out).sum(axis=self.axis, keepdims=True)
        return (out * (g - sum_gy),)


class LogSoftmax(Function):
    def forward(self, a, *, axis=-1):
        self.axis = axis
        mx = a.max(axis=axis, keepdims=True)
        shifted = a - mx
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out = shifted - lse
        self.save_for_backward(out)
        return out.astype(a.dtype)

    def backward(self, g):
        (out,) = self.saved
        # d logsoftmax / dx: g - softmax * g.sum(axis)
        sm = np.exp(out)
        return (g - sm * g.sum(axis=self.axis, keepdims=True),)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------


class MSELoss(Function):
    """Mean squared error (scalar output)."""

    def forward(self, pred, target):
        diff = pred - target
        self.save_for_backward(diff)
        self.n = diff.size
        return np.asarray((diff * diff).mean(), dtype=pred.dtype)

    def backward(self, g):
        (diff,) = self.saved
        grad = g * 2.0 * diff / self.n
        return grad, -grad


class BCELoss(Function):
    """Binary cross-entropy from probabilities in (0,1). Scalar output."""

    def forward(self, pred, target):
        eps = 1e-7
        p = np.clip(pred, eps, 1 - eps)
        self.save_for_backward(p, target)
        self.n = p.size
        loss = -(target * np.log(p) + (1 - target) * np.log(1 - p)).mean()
        return np.asarray(loss, dtype=pred.dtype)

    def backward(self, g):
        p, target = self.saved
        grad_p = g * (p - target) / (p * (1 - p) * self.n)
        # target typically has no grad, but produce anyway
        grad_t = g * (np.log(1 - p) - np.log(p)) / self.n
        return grad_p, grad_t


class BCEWithLogitsLoss(Function):
    """Numerically stable BCE that takes logits. Scalar output."""

    def forward(self, logits, target):
        # stable: max(x,0) - x*t + log(1+exp(-|x|))
        m = np.maximum(logits, 0)
        loss = m - logits * target + np.log1p(np.exp(-np.abs(logits)))
        self.save_for_backward(logits, target)
        self.n = logits.size
        return np.asarray(loss.mean(), dtype=logits.dtype)

    def backward(self, g):
        logits, target = self.saved
        # dL/dlogits = sigmoid(logits) - target, averaged
        s = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        ).astype(logits.dtype)
        grad_logits = g * (s - target) / self.n
        grad_target = -g * logits / self.n
        return grad_logits, grad_target


class CrossEntropyLoss(Function):
    """Stable softmax+NLL combined. logits: (N, C); target: (N,) int64. Scalar output."""

    def forward(self, logits, target):
        # logits: (N, C), target: (N,) integer class indices
        N, C = logits.shape
        mx = logits.max(axis=-1, keepdims=True)
        shifted = logits - mx
        lse = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        logp = shifted - lse  # (N, C)
        nll = -logp[np.arange(N), target.astype(np.int64)]
        self.save_for_backward(logp, target)
        self.N = N
        self.C = C
        return np.asarray(nll.mean(), dtype=logits.dtype)

    def backward(self, g):
        logp, target = self.saved
        # softmax probs
        p = np.exp(logp)
        # subtract 1 on target class
        p[np.arange(self.N), target.astype(np.int64)] -= 1
        grad_logits = g * p / self.N
        return grad_logits, None  # no grad wrt target


# ---------------------------------------------------------------------------
# functional wrappers
# ---------------------------------------------------------------------------


def relu(x: Tensor) -> Tensor:
    return ReLU.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid.apply(x)


def tanh(x: Tensor) -> Tensor:
    return Tanh.apply(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return LeakyReLU.apply(x, negative_slope=negative_slope)


def gelu(x: Tensor) -> Tensor:
    return GELU.apply(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return Softmax.apply(x, axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return LogSoftmax.apply(x, axis=axis)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return MSELoss.apply(pred, target)


def bce_loss(pred: Tensor, target: Tensor) -> Tensor:
    return BCELoss.apply(pred, target)


def bce_with_logits_loss(logits: Tensor, target: Tensor) -> Tensor:
    return BCEWithLogitsLoss.apply(logits, target)


def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    return CrossEntropyLoss.apply(logits, target)


# re-export extra losses
from nanograd.nn.losses_extra import huber_loss, l1_loss  # noqa: E402,F401
