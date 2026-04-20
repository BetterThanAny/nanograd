"""L1 and smooth-L1 (Huber) losses."""
from __future__ import annotations

import numpy as np

from nanograd.function import Function
from nanograd.tensor import Tensor


class L1Loss(Function):
    def forward(self, pred, target):
        diff = pred - target
        self.save_for_backward(diff)
        self.n = diff.size
        return np.asarray(np.abs(diff).mean(), dtype=pred.dtype)

    def backward(self, g):
        (diff,) = self.saved
        grad = g * np.sign(diff) / self.n
        return grad, -grad


class HuberLoss(Function):
    def forward(self, pred, target, *, delta=1.0):
        self.delta = delta
        diff = pred - target
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        self.save_for_backward(diff, abs_diff)
        self.n = diff.size
        return np.asarray(loss.mean(), dtype=pred.dtype)

    def backward(self, g):
        diff, abs_diff = self.saved
        d = self.delta
        # grad: diff where |diff|<d, d*sign(diff) otherwise
        grad = np.where(abs_diff <= d, diff, d * np.sign(diff))
        grad = g * grad / self.n
        return grad, -grad


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return L1Loss.apply(pred, target)


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    return HuberLoss.apply(pred, target, delta=delta)
